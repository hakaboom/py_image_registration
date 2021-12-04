# -*- coding: utf-8 -*-
import cv2
import numpy as np
from baseImage import Image
from image_registration.keypoint_matching.kaze import KAZE
from typing import Tuple, List, Union
from image_registration.exceptions import (CreateExtractorError, NoEnoughPointsError,
                                           CudaSurfInputImageError)


class CUDA_SURF(KAZE):
    METHOD_NAME = "CUDA_SURF"
    # 方向不变性:True检测/False不检测
    UPRIGHT = True
    # 检测器仅保留其hessian大于hessianThreshold的要素,值越大,获得的关键点就越少
    HESSIAN_THRESHOLD = 400

    def __init__(self, threshold: Union[int, float] = 0.8, rgb: bool = True,
                 _hessianThreshold=400, _nOctaves=4, _nOctaveLayers=2, _extended=True, _upright=False):
        super(CUDA_SURF, self).__init__(threshold, rgb)
        # 初始化参数
        self.extractor_parameters = dict(
            _hessianThreshold=_hessianThreshold, _nOctaves=_nOctaves, _nOctaveLayers=_nOctaveLayers,
            _extended=_extended, _upright=_upright,
        )
        try:
            self.detector = cv2.cuda.SURF_CUDA_create(**self.extractor_parameters)
        except Exception:
            raise CreateExtractorError('create cuda_surf extractor error')

    def create_matcher(self):
        """
        创建特征点匹配器
        :return: Brute-force descriptor matcher
        """
        # matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
        matcher = cv2.cuda_DescriptorMatcher.createBFMatcher(cv2.NORM_L2)
        return matcher

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

        self._check_image_size(im_source)
        self._check_image_size(im_search)

        im_source.transform_gpu()
        im_search.transform_gpu()

        return im_source, im_search

    def _check_image_size(self, image: Image):
        """
            SURF匹配特征点时,无法处理长宽太小的图片
            https://stackoverflow.com/questions/42492060/surf-cuda-error-while-computing-descriptors-and-keypoints
            https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/surf.cuda.cpp#L151
        """
        # (9 + 6 * 0) << nOctaves-1

        def calc_size(octave, layer):
            HAAR_SIZE0 = 9
            HAAR_SIZE_INC = 6
            return (HAAR_SIZE0 + HAAR_SIZE_INC * layer) << octave

        min_size = int(calc_size(self.detector.nOctaves - 1, 0))
        layer_height = image.size[0] >> (self.detector.nOctaves - 1)
        layer_width = image.size[1] >> (self.detector.nOctaves - 1)
        min_margin = ((calc_size((self.detector.nOctaves - 1), 2) >> 1) >> (self.detector.nOctaves - 1)) + 1

        if image.size[0] - min_size < 0 or image.size[1] - min_size < 0:
            raise CudaSurfInputImageError('The image size({width}x{height}) does not conform to SURF_CUDA standard'.
                                          format(width=image.size[1], height=image.size[0]))
        if layer_height - 2 * min_margin < 0 or layer_width - 2 * min_margin < 0:
            raise CudaSurfInputImageError('The image size({width}x{height}) does not conform to SURF_CUDA standard'.
                                          format(width=image.size[1], height=image.size[0]))

    def get_rect_from_good_matches(self, im_source: Image, im_search: Image,
                                   kp_sch, des_sch: cv2.cuda_GpuMat,
                                   kp_src, des_src: cv2.cuda_GpuMat):
        # TODO: 增加kp_sch,des_sch的类型提示,以及范函数返回值的类型提示
        matches = self.match_keypoints(des_sch=des_sch, des_src=des_src)
        good = self.get_good_in_matches(matches)

        kp_sch = self.detector.downloadKeypoints(kp_sch)
        kp_src = self.detector.downloadKeypoints(kp_src)

        rect = self.extract_good_points(im_source, im_search, kp_sch, kp_src, good)
        return rect, matches, good

    def match_keypoints(self, des_sch: cv2.cuda_GpuMat, des_src: cv2.cuda_GpuMat) -> List[List[cv2.DMatch]]:
        """Match descriptors (特征值匹配)."""
        # 匹配两个图片中的特征点集，k=2表示每个特征点取出2个最匹配的对应点:
        matches = self.matcher.knnMatch(des_sch, des_src, 2)
        return matches

    def get_keypoints_and_descriptors(self, image: cv2.cuda_GpuMat) -> Tuple[List[cv2.KeyPoint], cv2.cuda_GpuMat]:
        """获取图像特征点和描述符."""
        keypoints, descriptors = self.detector.detectWithDescriptors(image, None)

        if keypoints.size()[0] < 2:
            raise NoEnoughPointsError('{} detect not enough feature points in input images'.format(self.METHOD_NAME))
        return keypoints, descriptors
