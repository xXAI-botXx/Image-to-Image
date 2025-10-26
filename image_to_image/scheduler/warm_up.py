"""
Module to define a WarmUp Scheduler which can have a adter scheduler.

Class:
- WarmUpScheduler

By Tobia Ippolito
"""
# ---------------------------
#        > Imports <
# ---------------------------
import torch



# ---------------------------
#       > Scheduler <
# ---------------------------
class WarmUpScheduler(object):
    """
    Implements a learning rate scheduler with an initial warm-up phase.

    After the warm-up phase, an optional 'after scheduler' can take over
    to continue adjusting the learning rate according to another schedule.
    """
    def __init__(self, start_lr, end_lr, optimizer, scheduler=None, step_duration=1000):
        """
        Init WarmUp Scheduler.

        Parameter:
        - start_lr (float): 
            Initial learning rate at the start of warm-up.
        - end_lr (float): 
            Final learning rate at the end of warm-up.
        - optimizer (torch.optim.Optimizer): 
            Optimizer whose learning rate will be updated.
        - scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 
            Scheduler to apply after warm-up.
        - step_duration (int): 
            Number of steps over which to increase the learning rate.
        """
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
        """
        Performs a single step in the scheduler. 
        During warm-up, linearly increases the learning rate. 
        After warm-up, delegates to the optional after-scheduler.
        """
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
        """
        Returns the most recently applied learning rate as a list.
        """
        if self.current_step < self.step_duration:
            return [float(self.lrs[self.current_step-1])]
        elif self.scheduler:
            return self.scheduler.get_last_lr()
        else:
            return [self.end_lr]


    def state_dict(self):
        """
        Returns a dictionary with the current step and after-scheduler state.
        """
        return {
            "current_step": self.current_step,
            "after_scheduler": self.scheduler.state_dict() if self.scheduler else None
        }


    def load_state_dict(self, state):
        """
        Loads the scheduler state from a given dictionary.

        Parameter:
        - state (dict): 
            Dictionary containing 'current_step' and optional 'after_scheduler' state.
        """
        self.current_step = state["current_step"]
        if self.scheduler and state["after_scheduler"]:
            self.scheduler.load_state_dict(state["after_scheduler"])
    

 

    



