from typing import Union
from collections import OrderedDict
from loguru import logger
import numpy as np
import cv2

from baseImage import IMAGE
from .exceptions import BaseError, CreateExtractorError
from .settings import CVSTRATEGY, CVPARAMS


class Findit(object):
    def __init__(self, threshold=None, rgb=False):
        self.threshold = threshold
        self.rgb = rgb
        self.match_methods = self.init_matching_methods()

    def find_best_result(self, im_source: Union[IMAGE, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                         im_search: Union[IMAGE, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                         threshold: Union[int, float] = None, rgb: bool = None):
        """
        使用CVSTRATEGY中的方法,依次运行。在im_source中,找到最符合im_search的范围坐标
        :param im_source: 待匹配图像
        :param im_search: 图片模板
        :param threshold: 识别阈值(0~1)
        :param rgb: 是否使用rgb通道进行校验
        :return: 如果找到了,则会返回一个Rect对象。没找到则会返回None
        """
        # 初始化参数
        threshold = threshold is None and self.threshold or threshold
        rgb = rgb is None and self.rgb or rgb

        for method_name, method in self.match_methods.items():
            if isinstance(method, list):
                for func in method:
                    result = self._try_find_best(func, im_source=im_source, im_search=im_search,
                                                 threshold=threshold, rgb=rgb)
                    if result:
                        return result
            else:
                result = self._try_find_best(method, im_source=im_source, im_search=im_search,
                                             threshold=threshold, rgb=rgb)
                if result:
                    return result

        return None

    def find_all_results(self, im_source: Union[IMAGE, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                         im_search: Union[IMAGE, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                         threshold: Union[int, float] = None, max_count: int = 10, rgb: bool = None):
        """
        通过特征点匹配,在im_source中,找到符合im_search的范围坐标集合
        :param im_source: 待匹配图像
        :param im_search: 图片模板
        :param max_count: 最多可以返回的匹配数量
        :param threshold: 识别阈值(0~1)
        :param rgb: 是否使用rgb通道进行校验
        :return: 如果找到了,则会返回一个Rect对象。没找到则会返回None
        """
        threshold = threshold is None and self.threshold or threshold
        rgb = rgb is None and self.rgb or rgb

        for method_name, method in self.match_methods.items():
            if isinstance(method, list):
                for func in method:
                    result = self._try_find_all(func, im_source=im_source, im_search=im_search,
                                                threshold=threshold, rgb=rgb, max_count=max_count)
                    if result:
                        return result
            else:
                result = self._try_find_all(method, im_source=im_source, im_search=im_search,
                                            threshold=threshold, rgb=rgb, max_count=max_count)
                if result:
                    return result

        return None

    @staticmethod
    def _try_find_best(func, im_source, im_search, threshold, rgb):
        """
        尝试运行find_best_result,返回找到的坐标
        """
        try:
            match_result = func.find_best_result(im_source=im_source, im_search=im_search, threshold=threshold, rgb=rgb)
        except BaseError as err:
            logger.error('{} param:{}', err, func.get_extractor_parameters())
            return None
        else:
            return match_result

    @staticmethod
    def _try_find_all(func, im_source, im_search, threshold, rgb, max_count):
        """
        尝试运行find_best_result,返回找到的坐标
        """
        try:
            match_result = func.find_all_results(im_source=im_source, im_search=im_search,
                                                 threshold=threshold, rgb=rgb, max_count=max_count)
        except BaseError as err:
            logger.error('{} param:{}', err, func.get_extractor_parameters())
            return None
        else:
            return match_result

    @staticmethod
    def init_matching_methods():
        MATCH_METHODS = OrderedDict()

        for method in CVSTRATEGY:
            if method.__name__ in CVPARAMS:
                param_config = CVPARAMS[method.__name__]
            else:
                param_config = None

            if isinstance(param_config, list):
                MATCH_METHODS[method.__name__] = []
                for cfg in param_config:
                    func = Findit._create_extractor(method, cfg)
                    if func:
                        MATCH_METHODS[method.__name__].append(func)
            elif isinstance(param_config, dict):
                func = Findit._create_extractor(method, param_config)
                if func:
                    MATCH_METHODS[method.__name__] = func
            else:
                func = Findit._create_extractor(method)
                if func:
                    MATCH_METHODS[method.__name__] = func

        return MATCH_METHODS

    @staticmethod
    def _create_extractor(method, config=None):
        try:
            if config:
                func = method(**config)
            else:
                func = method()
        except CreateExtractorError:
            logger.error('create {} params=({}) error', method.__name__, config)
            return None
        else:
            return func
