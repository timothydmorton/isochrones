__all__ = ["getLogger", "initLogging"]

import sys
import logging


def getLogger():
    return logging.getLogger("isochrones")


def initLogging(filename, logger):
    if logger is None:
        logger = getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')

    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    logger.addHandler(sh)
    return logger
