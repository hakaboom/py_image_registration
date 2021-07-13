# -*- coding: utf-8 -*-
import cv2
from typing import Union
from image_registration.keypoint_matching.base import KAZE
from image_registration.exceptions import (CreateExtractorError)


class SURF(KAZE):
    METHOD_NAME = "SURF"

    def __init__(self, threshold: Union[int, float] = 0.8, rgb: bool = True,
                 hessianThreshold: int = 400, nOctaves: int = 4, nOctaveLayers: int = 3,
                 extended: bool = True, upright: bool = False):
        super(SURF, self).__init__(threshold, rgb)
        # 初始化参数
        self.extractor_parameters = dict(
            hessianThreshold=hessianThreshold, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
            extended=extended, upright=upright,
        )
        try:
            self.detector = cv2.xfeatures2d.SURF_create(**self.extractor_parameters)
        except Exception:
            raise CreateExtractorError('create surf extractor error')