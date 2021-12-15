# -*- coding: utf-8 -*-
import cv2
import numpy as np
from typing import Union, Tuple, List
from baseImage import Image
from image_registration.keypoint_matching.kaze import KAZE
from image_registration.exceptions import (CreateExtractorError, CudaOrbDetectorError,
                                           NoEnoughPointsError)


class CUDA_ORB(KAZE):
    METHOD_NAME = 'CUDA_ORB'

    def __init__(self, threshold: Union[int, float] = 0.8, rgb: bool = True,
                 nfeatures: int = 50000, scaleFactor: Union[int, float] = 1.2, nlevels: int = 8,
                 edgeThreshold: int = 31, firstLevel: int = 0, WTA_K: int = 2,
                 scoreType: int = cv2.ORB_HARRIS_SCORE, patchSize: int = 31, fastThreshold: int = 20,
                 blurForDescriptor: bool = False):
        super(CUDA_ORB, self).__init__(threshold, rgb)
        # 初始化参数
        self.extractor_parameters = dict(
            nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels,
            edgeThreshold=edgeThreshold, firstLevel=firstLevel, WTA_K=WTA_K,
            scoreType=scoreType, patchSize=patchSize, fastThreshold=fastThreshold,
            blurForDescriptor=blurForDescriptor,
        )
        try:
            # 创建ORB实例
            self.detector = cv2.cuda_ORB.create(**self.extractor_parameters)
        except Exception:
            raise CreateExtractorError('create cuda_orb extractor error')

    def check_image_input(self, im_source: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                          im_search: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes]) -> Tuple[Image, Image]:
        """
        检测输入的图像数据是否正确,并且转换为Gpu模式
        :param im_source: 待匹配图像
        :param im_search: 图片模板
        :return im_source, im_search
        """
        if not isinstance(im_source, Image):
            im_source = Image(im_source)
        if not isinstance(im_search, Image):
            im_search = Image(im_search)

        im_source.transform_gpu()
        im_search.transform_gpu()
        return im_source, im_search

    def create_matcher(self):
        """
        创建特征点匹配器
        :return: Brute-force descriptor matcher
        """
        matcher = cv2.cuda_DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING)
        return matcher

    def get_keypoints_and_descriptors(self, image: cv2.cuda_GpuMat) -> Tuple[list, cv2.cuda_GpuMat]:
        # https://github.com/prismai/opencv_contrib/commit/d7d6360fceb5881d596be95b03568d4dcdb7236d#diff-122b9c09d35cd89b7cee1eeb66189e4820f5663bbeda844908f05bf730c93e49
        try:
            keypoints, descriptors = self.detector.detectAndComputeAsync(image, None)
        except cv2.error:
            # https://github.com/opencv/opencv/issues/10573
            raise CudaOrbDetectorError('{} detect error, Try adjust detector params'.format(self.METHOD_NAME))
        else:
            keypoints = self.detector.convert(keypoints)

        if len(keypoints) < 2:
            raise NoEnoughPointsError('{} detect not enough feature points in input images'.format(self.METHOD_NAME))
        return keypoints, descriptors

    def match_keypoints(self, des_sch: cv2.cuda_GpuMat, des_src: cv2.cuda_GpuMat) -> List[List[cv2.DMatch]]:
        """
        特征点匹配
        :param des_sch: 图片模板的特征点集
        :param des_src: 待匹配图像的特征点集
        :return: 返回一个列表,包含最匹配的对应点
        """
        # k=2表示每个特征点取出2个最匹配的对应点
        matches = self.matcher.knnMatch(des_sch, des_src, 2)
        return matches

    @staticmethod
    def delect_rect_descriptors(rect, kp, des):
        tl, br = rect.tl, rect.br
        kp = kp.copy()
        des = des.download()

        delect_list = tuple(kp.index(i) for i in kp if tl.x <= i.pt[0] <= br.x and tl.y <= i.pt[1] <= br.y)
        for i in sorted(delect_list, reverse=True):
            kp.pop(i)

        des = np.delete(des, delect_list, axis=0)
        mat = cv2.cuda_GpuMat()
        mat.upload(des)
        return kp, mat

    @staticmethod
    def delect_good_descriptors(good, kp, des):
        kp = kp.copy()
        des = des.download()

        delect_list = [i.trainIdx for i in good]
        for i in sorted(delect_list, reverse=True):
            kp.pop(i)

        des = np.delete(des, delect_list, axis=0)
        mat = cv2.cuda_GpuMat()
        mat.upload(des)
        return kp, mat
