# -*- coding: utf-8 -*-
from typing import Union
import cv2

from image_registration.keypoint_matching.base import KAZE
from image_registration.exceptions import (CreateExtractorError)


class AKAZE(KAZE):
    METHOD_NAME = "AKAZE"

    def __init__(self, threshold: Union[int, float] = 0.8, rgb: bool = True,
                 descriptor_type: int = cv2.AKAZE_DESCRIPTOR_MLDB,
                 descriptor_size: int = 0, descriptor_channels: int = 3,
                 _threshold: float = 0.001, nOctaves: int = 4, nOctaveLayers: int = 4,
                 diffusivity: int = cv2.KAZE_DIFF_PM_G2):
        super(AKAZE, self).__init__(threshold, rgb)
        self.extractor_parameters = dict(
            descriptor_type=descriptor_type, descriptor_size=descriptor_size, descriptor_channels=descriptor_channels,
            threshold=_threshold, diffusivity=diffusivity,
            nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
        )
        try:
            self.detector = cv2.AKAZE_create(**self.extractor_parameters)
        except Exception:
            raise CreateExtractorError('create akaze extractor error')

    def create_matcher(self) -> cv2.BFMatcher:
        matcher = cv2.BFMatcher_create(cv2.NORM_L1)
        return matcher
