"""
python setup.py sdist
twine upload dist/*
"""
import time

import cv2
from loguru import logger
from baseImage import IMAGE
from collections import OrderedDict
from image_registration import match_template, ORB, CUDA_ORB, RootSIFT, CUDA_SURF, SURF
from image_registration.exceptions import NoEnoughPointsError, CreateExtractorError, BaseError
from image_registration.utils import pprint
from typing import Union


CVSTRATEGY = [SURF]
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    CVSTRATEGY = [SURF]
CVPARAMS = {
    'SURF': dict(upright=True),
    'ORB': [
        dict(nfeatures=60000),
        dict(nfeatures=60000, scaleFactor=2, nlevels=2, firstLevel=1),
        dict(nfeatures=60000, scaleFactor=2, nlevels=4, firstLevel=2),
    ]
}


class Match(object):
    def __init__(self, threshold=None, rgb=False):
        self.threshold = threshold
        self.rgb = rgb
        self.match_methods = self.init_matching_methods()

    def find_best(self, im_source, im_search, threshold: Union[int, float] = 0.8, rgb: bool = True,):
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

    def find_all(self, im_source, im_search, threshold: Union[int, float] = 0.8, rgb: bool = True, max_count: int = 10):
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

    def _try_find_best(self, func, im_source, im_search, threshold, rgb):
        try:
            match_result = func.find_best(im_source=im_source, im_search=im_search, threshold=threshold, rgb=rgb)
        except BaseError as err:
            logger.error('{} param:{}', err, func.get_extractor_parameters())
            return None
        else:
            return match_result

    def _try_find_all(self, func, im_source, im_search, threshold, rgb, max_count):
        try:
            match_result = func.find_all(im_source=im_source, im_search=im_search,
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
                    func = Match._create_extractor(method, cfg)
                    if func:
                        MATCH_METHODS[method.__name__].append(func)
            elif isinstance(param_config, dict):
                func = Match._create_extractor(method, param_config)
                if func:
                    MATCH_METHODS[method.__name__] = func
            else:
                func = Match._create_extractor(method)
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
