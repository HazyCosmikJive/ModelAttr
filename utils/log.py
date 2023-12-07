import os
import sys
import time
import logging

from utils.dist import get_rank


def setup_logger(config, distributed_rank=0, test=False):
    logger = logging.getLogger(config.common.exp_tag)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)

    # add rank filter
    rank = get_rank()
    logger.addFilter(lambda record: rank == 0)

    formatter = logging.Formatter(
        '%(asctime)s-[%(filename)s-line:%(lineno)d]-%(levelname)s: %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if test:  # save test log to pred TODO: add test_tag to logge
        save_dir = os.path.join(config.common.predpath, "logs")
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = config.common.logpath
    
    if save_dir:
        x = time.localtime(time.time())
        if test:
            log_name = config.common.test_tag + time.strftime('%Y-%m-%d-%H-%M-%S', x) + '.log'
        else:
            log_name = time.strftime('%Y-%m-%d-%H-%M-%S', x) + '.log'
        fh = logging.FileHandler(os.path.join(save_dir, log_name), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger