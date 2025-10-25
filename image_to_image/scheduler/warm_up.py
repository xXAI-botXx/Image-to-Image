# ---------------------------
#        > Imports <
# ---------------------------
import torch
import torch.nn as nn



# ---------------------------
#       > Scheduler <
# ---------------------------
class WarmUpScheduler(object):
    def __init__(self, start_lr, end_lr, optimizer, scheduler=None, step_duration=1000):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.current_lr = start_lr
        self.step_duration = step_duration
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.current_step = 0

        self.lrs = torch.linspace(start_lr, end_lr, step_duration)

        # set initial LR
        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = start_lr
        else:
            print("[WARNING] No optomizer was given to the WarmUp Scheduler!")


    def step(self):
        if self.current_step < self.step_duration:
            self.current_lr = float(self.lrs[self.current_step])
            if self.optimizer:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
        else:
            if self.scheduler is not None:
                self.scheduler.step()

        self.current_step += 1


    def get_last_lr(self):
        if self.current_step < self.step_duration:
            return [float(self.lrs[self.current_step-1])]
        elif self.scheduler:
            return self.scheduler.get_last_lr()
        else:
            return [self.end_lr]


    def state_dict(self):
        return {
            "current_step": self.current_step,
            "after_scheduler": self.scheduler.state_dict() if self.scheduler else None
        }


    def load_state_dict(self, state):
        self.current_step = state["current_step"]
        if self.scheduler and state["after_scheduler"]:
            self.scheduler.load_state_dict(state["after_scheduler"])
    

 

    



