# -*- coding: utf-8 -*-
import cv2
import numpy as np
from typing import Tuple, List, Union
from image_registration.keypoint_matching.kaze import KAZE
from image_registration.exceptions import (CreateExtractorError, NoModuleError, NoEnoughPointsError)


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
        try:
            # 创建ORB实例
            self.detector = cv2.ORB_create(**self.extractor_parameters)
        except Exception:
            raise CreateExtractorError('create orb extractor error')
        else:
            try:
                # https://docs.opencv.org/master/d7/d99/classcv_1_1xfeatures2d_1_1BEBLID.html
                # https://github.com/iago-suarez/beblid-opencv-demo
                self.descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
            except AttributeError:
                raise NoModuleError

    def create_matcher(self) -> cv2.DescriptorMatcher_create:
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        return matcher

    def get_good_in_matches(self, matches: list) -> List[cv2.DMatch]:
        """
        特征点过滤
        :param matches: 特征点集
        """
        good = []
        # 出现过matches对中只有1个参数的情况,会导致遍历的时候造成报错
        for v in matches:
            if len(v) == 2:
                if v[0].distance < self.FILTER_RATIO * v[1].distance:
                    good.append(v[0])
        return good

    def get_keypoints_and_descriptors(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        获取图像关键点(keypoints)与描述符(descriptors)
        :param image: 待检测的灰度图像
        :raise NoEnoughPointsError: 检测特征点数量少于2时,弹出异常
        :return: 关键点(keypoints)与描述符(descriptors)
        """
        keypoints = self.detector.detect(image, None)
        keypoints, descriptors = self.descriptor.compute(image, keypoints)

        if len(keypoints) < 2:
            raise NoEnoughPointsError('{} detect not enough feature points in input images'.format(self.METHOD_NAME))
        return keypoints, descriptors
