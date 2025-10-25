# ---------------------------
#        > Imports <
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia



# ---------------------------
#   > Loss Implementation <
# ---------------------------
class WeightedCombinedLoss(nn.Module):
    def __init__(self, 
                 silog_lambda=0.5, 
                 weight_silog=0.5, 
                 weight_grad=10.0, 
                 weight_ssim=5.0,
                 weight_edge_aware=10.0,
                 weight_l1=1.0,
                 weight_var=1.0,
                 weight_range=1.0,
                 weight_blur=1.0):
        super().__init__()
        self.silog_lambda = silog_lambda
        self.weight_silog = weight_silog
        self.weight_grad = weight_grad
        self.weight_ssim = weight_ssim
        self.weight_edge_aware = weight_edge_aware
        self.weight_l1 = weight_l1
        self.weight_var = weight_var
        self.weight_range = weight_range
        self.weight_blur = weight_blur

        self.avg_loss_silog = 0
        self.avg_loss_grad = 0
        self.avg_loss_ssim = 0
        self.avg_loss_l1 = 0
        self.avg_loss_edge_aware = 0
        self.avg_loss_var = 0
        self.avg_loss_range = 0
        self.avg_loss_blur = 0
        self.steps = 0

        # Instantiate SSIMLoss module
        self.ssim_module = kornia.losses.SSIMLoss(window_size=11, reduction='mean')
        # self.ssim_module = kornia.losses.MS_SSIMLoss(reduction='mean')


    def silog_loss(self, pred, target, weight_map):
        eps = 1e-6
        pred = torch.clamp(pred, min=eps)
        target = torch.clamp(target, min=eps)
        
        diff_log = torch.log(target) - torch.log(pred)
        diff_log = diff_log * weight_map

        loss = torch.sqrt(torch.mean(diff_log ** 2) -
                          self.silog_lambda * torch.mean(diff_log) ** 2)
        return loss

    def gradient_l1_loss(self, pred, target, weight_map):
        # Create Channel Dimension
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if weight_map.ndim == 3:
            weight_map = weight_map.unsqueeze(1)

        # Gradient in x-direction (horizontal -> dim=3)
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]

        # Gradient in y-direction (vertical -> dim=2)
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        weight_x = weight_map[:, :, :, 1:] * weight_map[:, :, :, :-1]
        weight_y = weight_map[:, :, 1:, :] * weight_map[:, :, :-1, :]

        loss_x = torch.mean(torch.abs(pred_grad_x - target_grad_x) * weight_x)
        loss_y = torch.mean(torch.abs(pred_grad_y - target_grad_y) * weight_y)
        
        # loss_x = F.l1_loss(pred_grad_x, target_grad_x) 
        # loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y

    def ssim_loss(self, pred, target, weight_map):
        # SSIM returns similarity, so we subtract from 1
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        # self.ssim_module = self.ssim_module.to(pred.device)
        return self.ssim_module(pred, target)

    def edge_aware_loss(self, pred, target, weight_map):
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if weight_map.ndim == 3:
            weight_map = weight_map.unsqueeze(1)

        pred_grad_x = pred[:, :, :, :-1] - pred[:, :, :, 1:]
        pred_grad_y = pred[:, :, :-1, :] - pred[:, :, 1:, :]

        target_grad_x = torch.mean(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:]), 1, keepdim=True)
        target_grad_y = torch.mean(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]), 1, keepdim=True)

        weight_x = weight_map[:, :, :, 1:] * weight_map[:, :, :, :-1]
        weight_y = weight_map[:, :, 1:, :] * weight_map[:, :, :-1, :]

        pred_grad_x *= torch.exp(-target_grad_x* weight_x) 
        pred_grad_y *= torch.exp(-target_grad_y* weight_y)

        # return (pred_grad_y.abs().mean() + target_grad_y.abs().mean())
        return (pred_grad_x.abs().mean() + pred_grad_y.abs().mean())

    def l1_loss(self, pred, target, weight_map):
        loss = torch.abs(target - pred) * weight_map
        return loss.mean()

    def variance_loss(self, pred, target):
        pred_var = torch.var(pred)
        target_var = torch.var(target)
        return F.mse_loss(pred_var, target_var)
    
    def range_loss(self, pred, target):
        pred_min, pred_max = torch.min(pred), torch.max(pred)
        target_min, target_max = torch.min(target), torch.max(target)
        
        min_loss = F.mse_loss(pred_min, target_min)
        max_loss = F.mse_loss(pred_max, target_max)
        
        return min_loss + max_loss

    def blur_loss(self, pred, target):
        laplacian_kernel = torch.tensor([[[[0, 1, 0],
                                           [1, -4, 1],
                                           [0, 1, 0]]]], dtype=pred.dtype, device=pred.device)

        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        pred_lap = F.conv2d(pred, laplacian_kernel, padding=1)
        target_lap = F.conv2d(target, laplacian_kernel, padding=1)

        return F.l1_loss(pred_lap, target_lap)

    def blur_loss(self, pred, target):
        laplacian_kernel = torch.tensor([[[[0, 1, 0],
                                           [1, -4, 1],
                                           [0, 1, 0]]]], dtype=pred.dtype, device=pred.device)

        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        pred_lap = F.conv2d(pred, laplacian_kernel, padding=1)
        target_lap = F.conv2d(target, laplacian_kernel, padding=1)

        return F.l1_loss(pred_lap, target_lap)

    def forward(self, pred, target, weight_map=None, should_calc_weight_map=False):
        if type(weight_map) == type(None):
            if should_calc_weight_map:
                weight_map = calc_weight_map(target)
            else:
                # no mask/weight-map
                # FIXME -> right
                weight_map = torch.ones_like(pred)

        loss_silog = self.silog_loss(pred, target, weight_map)
        loss_grad = self.gradient_l1_loss(pred, target, weight_map)
        loss_ssim = self.ssim_loss(pred, target, weight_map)
        loss_l1 = self.l1_loss(pred, target, weight_map)
        loss_edge_aware = self.edge_aware_loss(pred, target, weight_map)
        loss_var = self.variance_loss(pred, target)
        loss_range = self.range_loss(pred, target)
        loss_blur = self.blur_loss(pred, target)

        # reset avgs
        if self.steps > 24:
            self.step()

        self.avg_loss_silog += loss_silog
        self.avg_loss_grad += loss_grad
        self.avg_loss_ssim += loss_ssim
        self.avg_loss_l1 += loss_l1
        self.avg_loss_edge_aware += loss_edge_aware
        self.avg_loss_var += loss_var
        self.avg_loss_range += loss_range
        self.avg_loss_blur += loss_blur
        self.steps += 1

        total_loss = (
            self.weight_silog * loss_silog +
            self.weight_grad * loss_grad +
            self.weight_ssim * loss_ssim +
            self.weight_edge_aware * loss_edge_aware +
            self.weight_l1 * loss_l1 +
            self.weight_var * loss_var +
            self.weight_range * loss_range +
            self.weight_blur * loss_blur
        )

        return total_loss

    def step(self, epoch=None):
        self.avg_loss_silog = 0
        self.avg_loss_grad = 0
        self.avg_loss_ssim = 0
        self.avg_loss_l1 = 0
        self.avg_loss_edge_aware = 0
        self.avg_loss_var = 0
        self.avg_loss_range = 0
        self.avg_loss_blur = 0
        self.steps = 0

    def get_avg_losses(self):
        return (self.avg_loss_silog/self.steps,
                self.avg_loss_grad/self.steps,
                self.avg_loss_ssim/self.steps,
                self.avg_loss_l1/self.steps,
                self.avg_loss_edge_aware/self.steps,
                self.avg_loss_var/self.steps,
                self.avg_loss_range/self.steps,
                self.avg_loss_blur/self.steps
               )

    def get_dict(self):
        loss_silog, loss_grad, loss_ssim, loss_l1, loss_edge_aware, loss_var, loss_range, loss_blur = self.get_avg_losses()
        return {
                f"loss_silog": loss_silog, 
                f"loss_grad": loss_grad, 
                f"loss_ssim": loss_ssim,
                f"loss_L1": loss_l1,
                f"loss_edge aware": loss_edge_aware,
                f"loss_var": loss_var,
                f"loss_range": loss_range,
                f"loss_blur": loss_blur,
                f"weight_loss_silog": self.weight_silog, 
                f"weight_loss_grad": self.weight_grad,
                f"_weight_loss_ssim": self.weight_ssim,
                f"_weight_loss_L1": self.weight_l1,
                f"weight_loss_edge_aware": self.weight_edge_aware,
                f"weight_loss_var": self.weight_var,
                f"weight_loss_range": self.weight_range,
                f"weight_loss_blur": self.weight_blur
               }

def calc_weight_map(target):
    values, counts = torch.unique(target.flatten(), return_counts=True)
    all_counts = counts.sum().float()
    
    # weight_factor = 2.0
    # weights = {values[idx].item(): max(torch.exp( ( (1-(counts[idx].item()/all_counts))) *weight_factor), 0.0001) for idx in range(len(values))}
    
    weights = {values[idx].item(): 255.0/counts[idx].item() for idx in range(len(values))}

    # print(f"Weights:")
    # for cur_value, cur_counts in list(sorted(weights.items(), key=lambda x:x[0])):
    #     print('    - '+str(round(cur_value, 4))+': '+str(cur_counts.item()))

    weights_map = torch.zeros_like(target, dtype=torch.float)
    for cur_value in values:
        cur_value = cur_value.item()
        weights_map[target == cur_value] = weights[cur_value]

    return weights_map


