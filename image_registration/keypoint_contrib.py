#! usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy
from .keypoint_matching import KeypointMatch
from baseImage import IMAGE
from .exceptions import (CreateExtractorError, NoModuleError, NoEnoughPointsError,
                         CudaSurfInputImageError, CudaOrbDetectorError)
from typing import Tuple, List


class ORB(KeypointMatch):
    METHOD_NAME = "ORB"

    def __init__(self, threshold=0.8, *args, **kwargs):
        super(ORB, self).__init__(threshold)
        # 初始化参数
        kwargs = kwargs.copy()
        kwargs['nfeatures'] = kwargs.pop('nfeatures', 50000)

        try:
            # 创建ORB实例
            self.detector = cv2.ORB_create(*args, **kwargs)
        except Exception:
            raise CreateExtractorError('create orb extractor error')
        else:
            try:
                # https://docs.opencv.org/master/d7/d99/classcv_1_1xfeatures2d_1_1BEBLID.html
                # https://github.com/iago-suarez/beblid-opencv-demo
                self.descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
            except AttributeError:
                raise NoModuleError

    def create_matcher(self) -> cv2.DescriptorMatcher:
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        return matcher

    def get_good_in_matches(self, matches) -> List[cv2.DMatch]:
        good = []
        # 出现过matches对中只有1个参数的情况,会导致遍历的时候造成报错
        for v in matches:
            if len(v) == 2:
                if v[0].distance < self.FILTER_RATIO * v[1].distance:
                    good.append(v[0])
        return good

    def get_keypoints_and_descriptors(self, image: numpy.ndarray) -> Tuple[List[cv2.KeyPoint], numpy.ndarray]:
        keypoints = self.detector.detect(image, None)
        keypoints, descriptors = self.descriptor.compute(image, keypoints)

        if len(keypoints) < 2:
            raise NoEnoughPointsError('detect not enough feature points in input images')
        return keypoints, descriptors


class SIFT(KeypointMatch):
    METHOD_NAME = "SIFT"
    # SIFT识别特征点匹配，参数设置:
    FLANN_INDEX_KDTREE = 0

    def __init__(self, threshold=0.8, *args, **kwargs):
        super(SIFT, self).__init__(threshold)
        # 初始化参数
        kwargs = kwargs.copy()
        kwargs['edgeThreshold'] = kwargs.pop('edgeThreshold', 10)
        # 创建SIFT实例
        try:
            self.detector = cv2.SIFT_create(*args, **kwargs)
        except Exception:
            raise CreateExtractorError('create sift extractor error')


class RootSIFT(SIFT):
    METHOD_NAME = 'RootSIFT'

    def get_keypoints_and_descriptors(self, image):
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        keypoints, descriptors = self.rootSIFT_compute(image, keypoints)

        if len(keypoints) < 2:
            raise NoEnoughPointsError('detect not enough feature points in input images')
        return keypoints, descriptors

    def rootSIFT_compute(self, image, kps, eps=1e-7):
        keypoints, descriptors = self.detector.compute(image, kps)

        if len(keypoints) == 0:
            return [], None

        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        descriptors = numpy.sqrt(descriptors)

        return keypoints, descriptors


class SURF(KeypointMatch):
    # https://docs.opencv.org/master/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html
    METHOD_NAME = "SURF"
    # 方向不变性:0检测/1不检测
    UPRIGHT = 0
    # 检测器仅保留其hessian大于hessianThreshold的要素,值越大,获得的关键点就越少
    HESSIAN_THRESHOLD = 400
    # SURF识别特征点匹配:
    FLANN_INDEX_KDTREE = 0

    def __init__(self, threshold=0.8, *args, **kwargs):
        super(SURF, self).__init__(threshold)
        # 初始化参数
        kwargs = kwargs.copy()
        kwargs['hessianThreshold'] = kwargs.pop('hessianThreshold', self.HESSIAN_THRESHOLD)
        kwargs['upright '] = kwargs.pop('upright ', self.UPRIGHT)

        try:
            self.detector = cv2.xfeatures2d.SURF_create(*args, **kwargs)
        except Exception:
            raise CreateExtractorError('create surf extractor error')


class BRIEF(KeypointMatch):
    METHOD_NAME = "BRIEF"

    def __init__(self, threshold=0.8):
        super(BRIEF, self).__init__(threshold)
        # Initiate FAST detector
        self.star = cv2.xfeatures2d.StarDetector_create()
        # Initiate BRIEF extractor
        self.detector = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    def create_matcher(self) -> cv2.BFMatcher:
        matcher = cv2.BFMatcher_create(cv2.NORM_L1)
        return matcher

    def get_keypoints_and_descriptors(self, image: numpy.ndarray):
        # find the keypoints with STAR
        kp = self.star.detect(image, None)
        # compute the descriptors with BRIEF
        keypoints, descriptors = self.detector.compute(image, kp)

        if len(keypoints) < 2:
            raise NoEnoughPointsError('detect not enough feature points in input images')
        return keypoints, descriptors


class AKAZE(KeypointMatch):
    METHOD_NAME = "AKAZE"

    def __init__(self, threshold=0.8, *args, **kwargs):
        super(AKAZE, self).__init__(threshold)
        # Initiate AKAZE detector

        try:
            self.detector = cv2.AKAZE_create(*args, **kwargs)
        except Exception:
            raise CreateExtractorError('create akaze extractor error')

    def create_matcher(self) -> cv2.BFMatcher:
        matcher = cv2.BFMatcher_create(cv2.NORM_L1)
        return matcher


class CUDA_SURF(KeypointMatch):
    # https://docs.opencv.org/master/db/d06/classcv_1_1cuda_1_1SURF__CUDA.html
    METHOD_NAME = 'CUDA_SURF'
    # 方向不变性:True检测/False不检测
    UPRIGHT = True
    # 检测器仅保留其hessian大于hessianThreshold的要素,值越大,获得的关键点就越少
    HESSIAN_THRESHOLD = 400
    # SURF识别特征点匹配:
    FLANN_INDEX_KDTREE = 0

    def __init__(self, threshold=0.8, *args, **kwargs):
        super(CUDA_SURF, self).__init__(threshold)
        # 初始化参数
        kwargs = kwargs.copy()
        kwargs['_hessianThreshold'] = kwargs.pop('_hessianThreshold', self.HESSIAN_THRESHOLD)
        kwargs['_upright'] = kwargs.pop('_upright ', self.UPRIGHT)
        kwargs['_extended'] = kwargs.pop('_extended ', True)

        try:
            self.detector = cv2.cuda.SURF_CUDA_create(*args, **kwargs)
        except Exception:
            raise CreateExtractorError('create cuda_surf extractor error')

    def find_all(self, im_source, im_search, threshold=None):
        raise NotImplementedError

    def create_matcher(self):
        matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
        return matcher

    def check_detection_input(self, im_source: IMAGE, im_search: IMAGE):
        if not isinstance(im_source, IMAGE):
            im_source = IMAGE(im_source)
        if not isinstance(im_search, IMAGE):
            im_search = IMAGE(im_search)

        im_source.transform_gpu()
        im_search.transform_gpu()
        im_source, im_search = self.check_image_size(im_source, im_search)
        return im_source, im_search

    def check_image_size(self, im_source: IMAGE, im_search: IMAGE):
        try:
            self._check_image_size(im_source)
            self._check_image_size(im_search)
        except CudaSurfInputImageError:
            return None, None
        return im_source, im_search

    def _check_image_size(self, image: IMAGE):
        # https://stackoverflow.com/questions/42492060/surf-cuda-error-while-computing-descriptors-and-keypoints
        # https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/surf.cuda.cpp#L151
        # SURF匹配特征点时,无法处理长宽太小的图片
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
            raise CudaSurfInputImageError('{width}x{height}'.format(width=image.size[1], height=image.size[0]))
        if layer_height - 2 * min_margin < 0 or layer_width - 2 * min_margin < 0:
            raise CudaSurfInputImageError('{width}x{height}'.format(width=image.size[1], height=image.size[0]))

    def get_rect_from_good_matches(self, im_source, im_search, kp_sch, des_sch, kp_src, des_src):
        matches = self.match_keypoints(des_sch=des_sch, des_src=des_src)
        good = self.get_good_in_matches(matches)

        kp_sch = self.detector.downloadKeypoints(kp_sch)
        kp_src = self.detector.downloadKeypoints(kp_src)

        rect = self.extract_good_points(im_source, im_search, kp_sch, kp_src, good)
        return rect, matches, good

    def get_good_in_matches(self, matches) -> List[cv2.DMatch]:
        good = []
        for m, n in matches:
            if m.distance < self.FILTER_RATIO * n.distance:
                good.append(m)
        return good

    def match_keypoints(self, des_sch: cv2.cuda_GpuMat, des_src: cv2.cuda_GpuMat) -> List[List[cv2.DMatch]]:
        """Match descriptors (特征值匹配)."""
        # 匹配两个图片中的特征点集，k=2表示每个特征点取出2个最匹配的对应点:
        matches = self.matcher.knnMatch(des_sch, des_src, 2)
        return matches

    def get_keypoints_and_descriptors(self, image: cv2.cuda_GpuMat) -> Tuple[List[cv2.KeyPoint], cv2.cuda_GpuMat]:
        """获取图像特征点和描述符."""
        keypoints, descriptors = self.detector.detectWithDescriptors(image, None)

        if keypoints.size()[0] < 2:
            raise NoEnoughPointsError('detect not enough feature points in input images')
        return keypoints, descriptors


class CUDA_ORB(KeypointMatch):
    """
    cuda_orb在图像大小太小时,在detect阶段会出现ROI报错
    测试后发现可以同构建detector时, 修改金字塔nlevels和首层firstLevel大小来修正这个问题
    """
    METHOD_NAME = 'CUDA_ORB'
    FILTER_RATIO = 0.59

    def __init__(self, threshold=0.8, *args, **kwargs):
        super(CUDA_ORB, self).__init__(threshold)
        # 初始化参数
        kwargs = kwargs.copy()
        kwargs['nfeatures'] = kwargs.pop('nfeatures', 50000)
        try:
            # 创建ORB实例
            self.detector = cv2.cuda_ORB.create(*args, **kwargs)
        except Exception:
            raise CreateExtractorError('create cuda_orb extractor error')

    def check_detection_input(self, im_source: IMAGE, im_search: IMAGE) -> Tuple[IMAGE, IMAGE]:
        if not isinstance(im_source, IMAGE):
            im_source = IMAGE(im_source)
        if not isinstance(im_search, IMAGE):
            im_search = IMAGE(im_search)

        im_source.transform_gpu()
        im_search.transform_gpu()
        return im_source, im_search

    def create_matcher(self):
        matcher = cv2.cuda_DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING)
        return matcher

    def get_keypoints_and_descriptors(self, image: cv2.cuda_GpuMat) -> Tuple[list, cv2.cuda_GpuMat]:
        # https://github.com/prismai/opencv_contrib/commit/d7d6360fceb5881d596be95b03568d4dcdb7236d#diff-122b9c09d35cd89b7cee1eeb66189e4820f5663bbeda844908f05bf730c93e49
        try:
            keypoints, descriptors = self.detector.detectAndComputeAsync(image, None)
        except cv2.error:
            raise CudaOrbDetectorError('adjust detector params')
        keypoints = self.detector.convert(keypoints)

        if len(keypoints) < 2:
            raise NoEnoughPointsError('detect not enough feature points in input images')
        return keypoints, descriptors

    def match_keypoints(self, des_sch: cv2.cuda_GpuMat, des_src: cv2.cuda_GpuMat) -> List[List[cv2.DMatch]]:
        """Match descriptors (特征值匹配)."""
        # 匹配两个图片中的特征点集，k=2表示每个特征点取出2个最匹配的对应点:
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

        des = numpy.delete(des, delect_list, axis=0)
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

        des = numpy.delete(des, delect_list, axis=0)
        mat = cv2.cuda_GpuMat()
        mat.upload(des)
        return kp, mat
