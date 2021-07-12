import time
import re
from functools import wraps
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


class print_run_time(object):
    def __init__(self):
        pass

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            start_time = time.time()
            ret = func(*args, **kwargs)
            logger.debug("{}() run time is {time:.2f}ms".format(func.__name__, time=(time.time() - start_time) * 1000))
            return ret
        return wrapped_function


def match_time_debug(func):
    def wrapper(self, *args, **kwargs):
        # args = (im_source, im_search, kp_sch, des_sch, kp_src, des_src)
        im_source, im_search, kp_sch, des_sch, kp_src, des_src = args
        # result = (rect, matches, good)
        result = func(self, *args, **kwargs)
        rect, matches, good = result
        # logger.debug('sch_keypoints={}, src_keypoints={}, matches={}, good={}',
        #              len(kp_sch), len(kp_src), len(matches), len(good))

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


def get_type(value):
    s = re.findall(r'<class \'(.+?)\'>', str(type(value)))
    if s:
        return s[0]
    else:
        raise ValueError('unknown error,can not get type: value={}, type={}'.format(value, type(value)))


def get_space(SpaceNum=1):
    return '\t'*SpaceNum


def pprint(*args):
    _str = []
    for index, value in enumerate(args):
        if isinstance(value, (dict, tuple, list)):
            _str.append('[{index}]({type}) = {value}\n'.format(index=index, value=_print(value),
                                                                     type=get_type(value)))
        else:
            _str.append('[{index}]({type}) = {value}\n'.format(index=index, value=value,
                                                                   type=get_type(value)))
    print(''.join(_str))


def _print(args, SpaceNum=1):
    _str = []
    SpaceNum += 1
    if isinstance(args, (tuple, list)):
        _str.append('')
        for index, value in enumerate(args):
            _str.append('{space}[{index}]({type}) = {value}'.format(index=index, value=_print(value, SpaceNum),
                                                                    type=get_type(value), space=get_space(SpaceNum)))
    elif isinstance(args, dict):
        _str.append('')
        for key, value in args.items():
            _str.append('{space}[{key}]({type}) = {value}'.format(key=key, value=_print(value,SpaceNum),
                                                                  type=get_type(value), space=get_space(SpaceNum)))
    else:
        _str.append(str(args))

    return '\n'.join(_str)

