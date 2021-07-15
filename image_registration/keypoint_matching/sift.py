# -*- coding: utf-8 -*-
from typing import Union

import cv2
import numpy as np

from image_registration.keypoint_matching.kaze import KAZE
from image_registration.exceptions import (CreateExtractorError, NoEnoughPointsError)


class SIFT(KAZE):
    METHOD_NAME = "SIFT"

    def __init__(self, threshold: Union[int, float] = 0.8, rgb: bool = True,
                 nfeature: int = 0, nOctaveLayers: int = 3, contrastThreshold: float = 0.04, edgeThreshold: int = 10,
                 sigma: Union[int, float] = 1.6, descriptorType: int = cv2.CV_32F):
        super(SIFT, self).__init__(threshold, rgb)
        # 初始化参数
        self.extractor_parameters = dict(
            nfeatures=nfeature, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold, sigma=sigma, descriptorType=descriptorType
        )
        # 创建SIFT实例
        try:
            self.detector = cv2.SIFT_create(**self.extractor_parameters)
        except Exception:
            raise CreateExtractorError('create sift extractor error')


class RootSIFT(SIFT):
    METHOD_NAME = 'RootSIFT'

    def get_keypoints_and_descriptors(self, image):
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        keypoints, descriptors = self.rootSIFT_compute(image, keypoints)

        if len(keypoints) < 2:
            raise NoEnoughPointsError('{} detect not enough feature points in input images'.format(self.METHOD_NAME))
        return keypoints, descriptors

    def rootSIFT_compute(self, image, kps, eps=1e-7):
        keypoints, descriptors = self.detector.compute(image, kps)

        if len(keypoints) == 0:
            return [], None

        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        descriptors = np.sqrt(descriptors)

        return keypoints, descriptors
