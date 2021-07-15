# -*- coding: utf-8 -*-
from typing import Union

import cv2
import numpy as np

from image_registration.keypoint_matching.kaze import KAZE
from image_registration.exceptions import (NoEnoughPointsError, NoModuleError)


class BRIEF(KAZE):
    METHOD_NAME = "BRIEF"

    def __init__(self, threshold: Union[int, float] = 0.8, rgb: bool = True):
        super(BRIEF, self).__init__(threshold, rgb)
        try:
            # Initiate FAST detector
            self.star = cv2.xfeatures2d.StarDetector_create()
            # Initiate BRIEF extractor
            self.detector = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        except AttributeError:
            raise NoModuleError

    def create_matcher(self) -> cv2.BFMatcher:
        matcher = cv2.BFMatcher_create(cv2.NORM_L1)
        return matcher

    def get_keypoints_and_descriptors(self, image: np.ndarray):
        # find the keypoints with STAR
        kp = self.star.detect(image, None)
        # compute the descriptors with BRIEF
        keypoints, descriptors = self.detector.compute(image, kp)

        if len(keypoints) < 2:
            raise NoEnoughPointsError('{} detect not enough feature points in input images'.format(self.METHOD_NAME))
        return keypoints, descriptors
