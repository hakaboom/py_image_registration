# -*- coding: utf-8 -*-
from image_registration.base import KAZE
from typing import Union
import cv2


class ORB(KAZE):
    METHOD_NAME = "ORB"

    def __init__(self, threshold: Union[int, float] = 0.8, rgb: bool = True,
                 nfeatures: int = 50000, scaleFactor: Union[int, float] = 1.2, nlevels: int = 8,
                 edgeThreshold: int = 31, firstLevel: int = 0, WTA_K: int = 2,
                 scoreType: int = cv2.ORB_HARRIS_SCORE, patchSize: int = 31, fastThreshold: int = 20):
        super(ORB, self).__init__(threshold, rgb)
        # 初始化参数
        self.extractor_parameters = dict(
            nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels,
            edgeThreshold=edgeThreshold, firstLevel=firstLevel, WTA_K=WTA_K,
            scoreType=scoreType, patchSize=patchSize, fastThreshold=fastThreshold,
        )