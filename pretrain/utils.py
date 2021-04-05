import logging

logger = logging.getLogger()


def set_log(log_file=None):
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file != None:
        logfile = logging.FileHandler(log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)

        
class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count