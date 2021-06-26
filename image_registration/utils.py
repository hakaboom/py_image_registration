import time
from loguru import logger
from baseImage import IMAGE
import cv2


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


def match_time_debug(func):
    def wrapper(self, *args, **kwargs):
        # args = (im_source, im_search, kp_sch, des_sch, kp_src, des_src)
        im_source, im_search, kp_sch, des_sch, kp_src, des_src = args
        # result = (rect, matches, good)
        result = func(self, *args, **kwargs)
        rect, matches, good = result
        logger.debug('sch_keypoints={}, src_keypoints={}, matches={}, good={}',
                     len(kp_sch), len(kp_src), len(matches), len(good))

        # im_source, im_search = im_source.clone().imread(), im_search.clone().imread()
        # cv2.namedWindow(str(len(good)), cv2.WINDOW_KEEPRATIO)
        # cv2.imshow(str(len(good)), cv2.drawMatches(im_search, kp_sch, im_source, kp_src, good, None, flags=2))
        # cv2.namedWindow(str(len(good) + 1), cv2.WINDOW_KEEPRATIO)
        # cv2.imshow(str(len(good) + 1), cv2.drawKeypoints(im_source, kp_src, im_source, color=(255, 0, 255)))
        # cv2.namedWindow(str(len(good) + 2), cv2.WINDOW_KEEPRATIO)
        # cv2.imshow(str(len(good) + 2), cv2.drawKeypoints(im_search, kp_sch, im_search, color=(255, 0, 255)))
        # cv2.waitKey(0)
        return result

    return wrapper


def print_best_result(func):
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        logger.debug('[{FUNC_NAME}({METHOD_NAME})]rect={Rect}, confidence={confidence}, time={time:.2f}ms'.format(
            FUNC_NAME=func.__name__, METHOD_NAME=self.METHOD_NAME,
            confidence=(result and result['confidence'] or None),
            Rect=(result and result['rect'] or None),
            time=(time.time() - start_time) * 1000))

        return result

    return wrapper


def print_all_result(func):
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        logger.debug('[{FUNC_NAME}({METHOD_NAME})] find counts:{counts}, time={time:.2f}ms{result}'.format(
            FUNC_NAME=func.__name__, METHOD_NAME=self.METHOD_NAME,
            counts=len(result), time=(time.time() - start_time) * 1000,
            result=''.join(['\n\t{}, confidence={}'.format(x['rect'], x['confidence']) for x in result])))

        return result

    return wrapper
