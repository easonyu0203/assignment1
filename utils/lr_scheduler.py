class LearningRateScheduler:
    def __init__(self, start_lr, end_lr, max_iters):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.max_iters = max_iters

    def get_lr(self, current_iter):
        return self.start_lr - (self.start_lr - self.end_lr) * (current_iter / self.max_iters)
