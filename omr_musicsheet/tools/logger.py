"""
"""
import logging


def init_logger(logger=None, level=logging.DEBUG):
    """

    :param logger:
    :param level:
    :return:
    """
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=level)
    if logger:
        logger.setLevel(level)
