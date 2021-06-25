import time
from loguru import logger


def generate_result(rect, confi):
    """Format the result: 定义图像识别结果格式."""
    ret = {
        'rect': rect,
        'confidence': confi,
    }
    return ret


def print_run_time(func):
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        ret = func(self, *args, **kwargs)
        logger.debug("{}() run time is {time:.2f}ms".format(func.__name__, time=(time.time() - start_time) * 1000))
        return ret

    return wrapper
