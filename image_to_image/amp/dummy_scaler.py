
class DummyScaler:
    """
    A dummy AMP scaler that does nothing.

    Can be used as a drop-in replacement for torch.cuda.amp.GradScaler.
    """

    def __init__(self):
        pass

    def scale(self, loss):
        # Just return the loss unchanged
        return loss

    def step(self, optimizer):
        # Do nothing
        optimizer.step()

    def update(self, new_scale=None):
        # Do nothing
        pass

    def unscale_(self, optimizer):
        # Do nothing
        pass

    def state_dict(self):
        # Return empty dict for compatibility
        return {}

    def load_state_dict(self, state_dict):
        # Do nothing
        pass



