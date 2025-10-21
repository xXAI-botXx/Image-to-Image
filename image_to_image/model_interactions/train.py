# ---------------------------
#        > Imports <
# ---------------------------
import os

import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision

# Experiment tracking
import mlflow
import mlflow.pytorch
from torch.utils.tensorboard import SummaryWriter


from ..utils.argument_parsing import parse_args
from ..data.physgen import PhysGenDataset
from ..models.resfcn import ResFCN 



# ---------------------------
#      > Train Helpers <
# ---------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    # change to train mode -> calc gradients
    model.train()

    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # reset gradients 
        optimizer.zero_grad()

        if scaler:
            with autocast():
                y_predict = model(x)
                loss = criterion(y_predict, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            y_predict = model(x)
            loss = criterion(y_predict, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)



@torch.no_grad()
def evaluate(model, loader, criterion, device, writer=None, epoch=None, save_path=None, cmap="gray"):
    model.eval()

    total_loss = 0.0
    

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_predict = model(x)
        total_loss += criterion(y_predict, y).item()


        if writer:
            # Convert to grid
            img_grid_input = torchvision.utils.make_grid(x[:4].cpu(), normalize=True, scale_each=True)
            img_grid_pred = torchvision.utils.make_grid(y_predict[:4].unsqueeze(1).float().cpu() / y_predict.max().item())
            img_grid_gt = torchvision.utils.make_grid(y[:4].unsqueeze(1).float().cpu() / y.max().item())

            # Log to TensorBoard
            writer.add_image("Input", img_grid_input, epoch)
            writer.add_image("Prediction", img_grid_pred, epoch)
            writer.add_image("GroundTruth", img_grid_gt, epoch)

        if save_path:
            plt.imsave(os.path.join(save_path, f"{epoch}_prediction.png"), 
                       y_predict[0].detach().cpu().numpy().squeeze(), cmap=cmap)
            plt.imsave(os.path.join(save_path, f"{epoch}_input.png"), 
                       x[0][0].detach().cpu().numpy().squeeze(), cmap=cmap)
            plt.imsave(os.path.join(save_path, f"{epoch}_ground_truth.png"), 
                       y[0].detach().cpu().numpy().squeeze(), cmap=cmap)

    return total_loss / len(loader)



# checkpoint helper
def save_checkpoint(model, optimizer, scheduler, epoch, path='ckpt.pth'):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'sched_state': scheduler.state_dict() if scheduler else None
    }, path)

# usage example in training script
# for epoch in range(start, epochs):
#     train_loss = train_one_epoch(..., scaler=scaler)
#     val_loss, val_acc = evaluate(...)
#     save_checkpoint(...)

# ---------------------------
#        > Train Main <
# ---------------------------
def train(args=None):
    # Parse arguments
    if args is None:
        args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset loading
    train_dataset = PhysGenDataset(variation=args.data_variation, mode="train", input_type=args.input_type, output_type=args.output_type, 
                                   fake_rgb_output=args.fake_rgb_output, make_14_dividable_size=args.make_14_dividable_size)
    val_dataset = PhysGenDataset(variation=args.data_variation, mode="eval", input_type=args.input_type, output_type=args.output_type, 
                                   fake_rgb_output=args.fake_rgb_output, make_14_dividable_size=args.make_14_dividable_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model Loading
    model = ResFCN(in_channels=args.resfcn_in_channels, hidden_channels=args.resfcn_hidden_channels, out_channels=args.resfcn_out_channels, num_blocks=args.resfcn_num_blocks).to(device)

    # Loss
    if args.loss == "l1":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"'{args.loss}' is not an supported loss.")

    # Optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"'{args.optimizer}' is not an supported optimizer.")

    # Scheduler
    if args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    else:
        raise ValueError(f"'{args.scheduler}' is not an supported scheduler.")

    # Scaler
    if args.scaler == None:
        scaler = None
    elif args.scaler == "grad":
        scaler = GradScaler()
    else:
        raise ValueError(f"'{args.scaler}' is not an supported scaler.")

    os.makedirs(args.save_dir, exist_ok=True)

    mlflow.set_experiment(args.experiment_name)
    # same as:
        # mlflow.create_experiment(args.experiment_name)
        # mlflow.get_experiment_by_name(experiment_name)

    # Start MLflow run
    with mlflow.start_run(run_name=args.run_name):

        mlflow.pytorch.log_model(model, args.model)

        # Log hyperparameters
        mlflow.log_params({
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "device": str(device),
            "variation": args.data_variation,
            "model": args.model
        })

        # TensorBoard writer
        writer = SummaryWriter(log_dir=args.tensorboard_path)

        # Run Training
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            if epoch % 10 == 0:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device, writer=writer, epoch=epoch, save_path=args.save_path, cmap=args.cmap)

            print(f"[Epoch {epoch}/{args.epochs}] Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Log to TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0]
            }, step=epoch)

            scheduler.step()

            # Checkpoint speichern
            if epoch % 10 == 0:
                ckeckpoint_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
                save_checkpoint(model, optimizer, scheduler, epoch, ckeckpoint_path)

                # Log model checkpoint path
                mlflow.log_artifact(ckeckpoint_path)

         # Log final model
        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.end_run()
        writer.close()

        print("Training completed.")



if __name__ == "__main__":
    train()



