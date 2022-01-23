# -*- coding: utf-8 -*-

from typing import Union, Tuple, List
import cv2
import numpy as np
import time
from baseImage import Image, Rect
from image_registration.template_matching.matchTemplate import MatchTemplate
from image_registration.exceptions import (NoEnoughPointsError, CreateExtractorError, PerspectiveTransformError,
                                           HomographyError, MatchResultError)
from image_registration.utils import generate_result, match_time_debug


class KAZE(object):
    FLANN_INDEX_KDTREE = 0
    FILTER_RATIO = 0.59
    METHOD_NAME = 'KAZE'
    template = MatchTemplate()

    def __init__(self, threshold: Union[int, float] = 0.8, rgb: bool = True):
        """
        初始化
        :param threshold: 识别阈值(0~1)
        :param rgb: 是否使用rgb通道进行校验
        :return: None
        """
        self.threshold = threshold
        self.rgb = rgb
        self.detector = cv2.KAZE_create()  # type: cv2.KAZE
        self.extractor_parameters = dict()  # TODO:增加相应参数
        try:
            self.matcher = self.create_matcher()
        except AttributeError:
            raise CreateExtractorError('create KAZE matcher error')

    def find_best_result(self, im_source: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                         im_search: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                         threshold: Union[int, float] = None, rgb: bool = None):
        """
        通过特征点匹配,在im_source中,找到最符合im_search的范围坐标
        :param im_source: 待匹配图像
        :param im_search: 图片模板
        :param threshold: 识别阈值(0~1)
        :param rgb: 是否使用rgb通道进行校验
        :return: 如果找到了,则会返回一个Rect对象。没找到则会返回None
        """
        # 初始化参数
        threshold = threshold is None and self.threshold or threshold
        rgb = rgb is None and self.rgb or rgb

        im_source, im_search = self.check_image_input(im_source=im_source, im_search=im_search)
        # 获取特征点
        kp_sch, des_sch = self.get_keypoints_and_descriptors(image=im_search.rgb_2_gray())
        kp_src, des_src = self.get_keypoints_and_descriptors(image=im_source.rgb_2_gray())

        # 在特征点集中,匹配最接近的特征点
        rect, matches, good = self.get_rect_from_good_matches(im_source, im_search, kp_sch, des_sch, kp_src, des_src)

        if not rect:
            return None
        # 根据识别的结果,从待匹配图像中截取范围,进行模板匹配求出相似度
        confidence = self._cal_confidence(im_source=im_source, im_search=im_search, crop_rect=rect, rgb=rgb)
        best_match = generate_result(rect=rect, confi=confidence)
        return best_match if confidence > threshold else None

    def find_all_results(self, im_source: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                         im_search: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                         threshold: Union[int, float] = None, max_count: int = 10, rgb: bool = None):
        """
        通过特征点匹配,在im_source中,找到符合im_search的范围坐标集合
        :param im_source: 待匹配图像
        :param im_search: 图片模板
        :param max_count: 最多可以返回的匹配数量
        :param threshold: 识别阈值(0~1)
        :param rgb: 是否使用rgb通道进行校验
        :return: 如果找到了,则会返回一个Rect对象。没找到则会返回None
        """
        threshold = threshold or self.threshold
        rgb = rgb is None and self.rgb or rgb

        im_source, im_search = self.check_image_input(im_source=im_source, im_search=im_search)
        result = []
        # 获取特征点
        kp_sch, des_sch = self.get_keypoints_and_descriptors(image=im_search.rgb_2_gray())
        kp_src, des_src = self.get_keypoints_and_descriptors(image=im_source.rgb_2_gray())

        # 1.0.17
        if type(kp_sch) == tuple:
            kp_sch = list(kp_sch)
        if type(kp_src) == tuple:
            kp_src = list(kp_src)

        while len(kp_src) > 2 and len(kp_sch) > 2:
            rect, matches, good = self.get_rect_from_good_matches(im_source, im_search,
                                                                  kp_sch, des_sch,
                                                                  kp_src, des_src)
            if not rect:
                break

            confidence = self._cal_confidence(im_source=im_source, im_search=im_search, crop_rect=rect, rgb=rgb)

            if confidence > threshold and len(result) < max_count:
                result.append(generate_result(rect, confidence))

            kp_src, des_src = self.delect_good_descriptors(good, kp_src, des_src)
            kp_src, des_src = self.delect_rect_descriptors(rect, kp_src, des_src)

        return result

    @staticmethod
    def check_image_input(im_source: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                          im_search: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes]) -> Tuple[Image, Image]:
        """
        检测输入的图像数据是否正确
        :param im_source: 待匹配图像
        :param im_search: 图片模板
        :return im_source, im_search
        """
        if not isinstance(im_source, Image):
            im_source = Image(im_source)
        if not isinstance(im_search, Image):
            im_search = Image(im_search)

        im_source.transform_cpu()
        im_search.transform_cpu()
        return im_source, im_search

    def create_matcher(self) -> cv2.FlannBasedMatcher:
        """
        创建特征点匹配器
        :return: FlannBasedMatcher
        """
        index_params = {'algorithm': self.FLANN_INDEX_KDTREE, 'tree': 5}
        # 指定递归遍历的次数. 值越高结果越准确，但是消耗的时间也越多
        search_params = {'checks': 50}
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        return matcher

    def get_keypoints_and_descriptors(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        获取图像关键点(keypoints)与描述符(descriptors)
        :param image: 待检测的灰度图像
        :raise NoEnoughPointsError: 检测特征点数量少于2时,弹出异常
        :return: 关键点(keypoints)与描述符(descriptors)
        """
        keypoints, descriptors = self.detector.detectAndCompute(image, None)

        if len(keypoints) < 2:
            raise NoEnoughPointsError('{} detect not enough feature points in input images'.format(self.METHOD_NAME))
        return keypoints, descriptors

    # @match_time_debug
    def get_rect_from_good_matches(self, im_source: Image, im_search: Image,
                                   kp_sch: List[cv2.KeyPoint], des_sch: np.ndarray,
                                   kp_src: List[cv2.KeyPoint], des_src: np.ndarray) \
            -> Tuple[Rect, List[List[cv2.DMatch]], List[cv2.DMatch]]:
        """ 从特征点里获取最佳的范围 """
        matches = self.match_keypoints(des_sch=des_sch, des_src=des_src)
        good = self.get_good_in_matches(matches)
        rect = self.extract_good_points(im_source, im_search, kp_sch, kp_src, good)
        return rect, matches, good

    def _cal_confidence(self, im_source, im_search, crop_rect: Rect, rgb: bool) -> Union[int, float]:
        """
        将截图和识别结果缩放到大小一致,并计算可信度
        :param im_source: 待匹配图像
        :param im_search: 图片模板
        :param crop_rect: 需要在im_source截取的区域
        :param rgb: 是否使用rgb通道进行校验
        :raise MatchResultError: rect范围超出了im_source边界
        :return: 返回可信度(0~1)
        """
        try:
            target_img = im_source.crop_image(crop_rect)
        except OverflowError:
            raise MatchResultError(f"Target area({crop_rect}) out of screen{im_source.size}")

        h, w = im_search.size
        target_img.resize(w, h)

        if rgb:
            confidence = self.template.cal_rgb_confidence(im_source=im_search, im_search=target_img)
        else:
            confidence = self.template.cal_ccoeff_confidence(im_source=im_search, im_search=target_img)

        confidence = (1 + confidence) / 2
        return confidence

    def match_keypoints(self, des_sch: np.ndarray, des_src: np.ndarray) -> List[List[cv2.DMatch]]:
        """
        特征点匹配
        :param des_sch: 图片模板的特征点集
        :param des_src: 待匹配图像的特征点集
        :return: 返回一个列表,包含最匹配的对应点
        """
        # k=2表示每个特征点取出2个最匹配的对应点
        matches = self.matcher.knnMatch(des_sch, des_src, 2)
        return matches

    def get_good_in_matches(self, matches: list) -> List[cv2.DMatch]:
        """
        特征点过滤
        :param matches: 特征点集
        """
        good = []
        for m, n in matches:
            if m.distance < self.FILTER_RATIO * n.distance:
                good.append(m)
        return good

    def extract_good_points(self, im_source: Image, im_search: Image,
                            kp_sch: List[cv2.KeyPoint], kp_src: List[cv2.KeyPoint], good: List[cv2.DMatch]):
        """
        根据匹配点(good)数量,提取识别区域
        """
        if len(good) in [0, 1]:
            # 匹配点数量太少,返回None
            return None
        elif len(good) in [2, 3]:
            if len(good) == 2:
                origin_result = self._handle_two_good_points(im_source=im_source, im_search=im_search,
                                                             kp_sch=kp_sch, kp_src=kp_src, good=good)
            else:
                origin_result = self._handle_three_good_points(im_source=im_source, im_search=im_search,
                                                               kp_sch=kp_sch, kp_src=kp_src, good=good)

            if isinstance(origin_result, dict):
                return None
            else:
                return origin_result
        else:
            # 匹配点数量>=4,使用单矩阵映射求出目标区域
            return self._many_good_pts(im_source=im_source, im_search=im_search,
                                       kp_sch=kp_sch, kp_src=kp_src, good=good)

    @staticmethod
    def _find_homography(sch_pts, src_pts):
        """
        多组特征点对时，求取单向性矩阵
        """
        try:
            M, mask = cv2.findHomography(sch_pts, src_pts, cv2.RANSAC, 5.0)
        except cv2.error:
            import traceback
            traceback.print_exc()
            raise HomographyError("OpenCV error in _find_homography()...")
        else:
            if mask is None:
                raise HomographyError("In _find_homography(), find no mask...")
            else:
                return M, mask

    def _handle_two_good_points(self, im_source: Image, im_search: Image,
                                kp_sch: List[cv2.KeyPoint], kp_src: List[cv2.KeyPoint], good: List[cv2.DMatch]):
        """ 处理两对特征点的情况 """
        pts_sch1 = int(kp_sch[good[0].queryIdx].pt[0]), int(kp_sch[good[0].queryIdx].pt[1])
        pts_sch2 = int(kp_sch[good[1].queryIdx].pt[0]), int(kp_sch[good[1].queryIdx].pt[1])
        pts_src1 = int(kp_src[good[0].trainIdx].pt[0]), int(kp_src[good[0].trainIdx].pt[1])
        pts_src2 = int(kp_src[good[1].trainIdx].pt[0]), int(kp_src[good[1].trainIdx].pt[1])

        return self._two_good_points(pts_sch1, pts_sch2, pts_src1, pts_src2, im_search, im_source)

    def _handle_three_good_points(self, im_source: Image, im_search: Image,
                                  kp_sch: List[cv2.KeyPoint], kp_src: List[cv2.KeyPoint], good: List[cv2.DMatch]):
        """ 处理三对特征点的情况 """
        # 拿出sch和src的两个点(点1)和(点2点3的中点)，
        # 然后根据两个点原则进行后处理(注意ke_sch和kp_src以及queryIdx和trainIdx):
        pts_sch1 = int(kp_sch[good[0].queryIdx].pt[0]), int(kp_sch[good[0].queryIdx].pt[1])
        pts_sch2 = int((kp_sch[good[1].queryIdx].pt[0] + kp_sch[good[2].queryIdx].pt[0]) / 2), int(
            (kp_sch[good[1].queryIdx].pt[1] + kp_sch[good[2].queryIdx].pt[1]) / 2)
        pts_src1 = int(kp_src[good[0].trainIdx].pt[0]), int(kp_src[good[0].trainIdx].pt[1])
        pts_src2 = int((kp_src[good[1].trainIdx].pt[0] + kp_src[good[2].trainIdx].pt[0]) / 2), int(
            (kp_src[good[1].trainIdx].pt[1] + kp_src[good[2].trainIdx].pt[1]) / 2)
        return self._two_good_points(pts_sch1, pts_sch2, pts_src1, pts_src2, im_search, im_source)

    @staticmethod
    def _two_good_points(pts_sch1, pts_sch2, pts_src1, pts_src2, im_search, im_source):
        """返回两对匹配特征点情形下的识别结果."""
        # 先算出中心点(在im_source中的坐标)：
        middle_point = [int((pts_src1[0] + pts_src2[0]) / 2), int((pts_src1[1] + pts_src2[1]) / 2)]
        pypts = []
        # 如果特征点同x轴或同y轴(无论src还是sch中)，均不能计算出目标矩形区域来，此时返回值同good=1情形
        if pts_sch1[0] == pts_sch2[0] or pts_sch1[1] == pts_sch2[1] or pts_src1[0] == pts_src2[0] or pts_src1[1] == \
                pts_src2[1]:
            confidence = 0.5
            return {'result': middle_point, 'rectangle': pypts, 'confidence': confidence}
        # 计算x,y轴的缩放比例：x_scale、y_scale，从middle点扩张出目标区域:(注意整数计算要转成浮点数结果!)
        h, w = im_search.size[:2]
        h_s, w_s = im_source.size[:2]
        x_scale = abs(1.0 * (pts_src2[0] - pts_src1[0]) / (pts_sch2[0] - pts_sch1[0]))
        y_scale = abs(1.0 * (pts_src2[1] - pts_src1[1]) / (pts_sch2[1] - pts_sch1[1]))
        # 得到scale后需要对middle_point进行校正，并非特征点中点，而是映射矩阵的中点。
        sch_middle_point = int((pts_sch1[0] + pts_sch2[0]) / 2), int((pts_sch1[1] + pts_sch2[1]) / 2)
        middle_point[0] = middle_point[0] - int((sch_middle_point[0] - w / 2) * x_scale)
        middle_point[1] = middle_point[1] - int((sch_middle_point[1] - h / 2) * y_scale)
        middle_point[0] = max(middle_point[0], 0)  # 超出左边界取0  (图像左上角坐标为0,0)
        middle_point[0] = min(middle_point[0], w_s - 1)  # 超出右边界取w_s-1
        middle_point[1] = max(middle_point[1], 0)  # 超出上边界取0
        middle_point[1] = min(middle_point[1], h_s - 1)  # 超出下边界取h_s-1

        # 计算出来rectangle角点的顺序：左上角->左下角->右下角->右上角， 注意：暂不考虑图片转动
        # 超出左边界取0, 超出右边界取w_s-1, 超出下边界取0, 超出上边界取h_s-1
        x_min, x_max = int(max(middle_point[0] - (w * x_scale) / 2, 0)), int(
            min(middle_point[0] + (w * x_scale) / 2, w_s - 1))
        y_min, y_max = int(max(middle_point[1] - (h * y_scale) / 2, 0)), int(
            min(middle_point[1] + (h * y_scale) / 2, h_s - 1))
        # 目标矩形的角点按左上、左下、右下、右上的点序：(x_min,y_min)(x_min,y_max)(x_max,y_max)(x_max,y_min)
        pts = np.float32([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]).reshape(-1, 1, 2)
        for npt in pts.astype(int).tolist():
            pypts.append(tuple(npt[0]))
        return Rect(x=x_min, y=y_min, width=(x_max - x_min), height=(y_max - y_min))

    def _many_good_pts(self, im_source: Image, im_search: Image, kp_sch: List[cv2.KeyPoint], kp_src: List[cv2.KeyPoint],
                       good: List[cv2.DMatch]) -> Rect:
        """
        特征点匹配数量>=4时,使用单矩阵映射,求出识别的目标区域
        :param im_source: 待匹配图像
        :param im_search: 图片模板
        :param kp_sch: im_search的关键点
        :param kp_src: im_source的关键点
        :param good: 筛选后的特征点集
        :return: 返回转换后的范围Rect
        :raise PerspectiveTransformError: cv透视变换时,出现异常
        """
        sch_pts, img_pts = np.float32([kp_sch[m.queryIdx].pt for m in good]).reshape(
            -1, 1, 2), np.float32([kp_src[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # M是转化矩阵
        M, mask = self._find_homography(sch_pts, img_pts)
        # 计算四个角矩阵变换后的坐标，也就是在大图中的目标区域的顶点坐标:
        h, w = im_search.shape[:2]
        h_s, w_s = im_source.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        try:
            dst = cv2.perspectiveTransform(pts, M)
        except cv2.error as err:
            raise PerspectiveTransformError(err)

        def cal_rect_pts(_dst):
            return [tuple(npt[0]) for npt in np.rint(_dst).astype(np.float).tolist()]

        pypts = cal_rect_pts(dst)
        # pypts四个值按照顺序分别是: 左上,左下,右下,右上
        # 注意：虽然4个角点有可能越出source图边界，但是(根据精确化映射单映射矩阵M线性机制)中点不会越出边界
        lt, br = pypts[0], pypts[2]
        # 考虑到算出的目标矩阵有可能是翻转的情况，必须进行一次处理，确保映射后的“左上角”在图片中也是左上角点：
        x_min, x_max = min(lt[0], br[0]), max(lt[0], br[0])
        y_min, y_max = min(lt[1], br[1]), max(lt[1], br[1])
        # 挑选出目标矩形区域可能会有越界情况，越界时直接将其置为边界：
        # 超出左边界取0，超出右边界取w_s-1，超出下边界取0，超出上边界取h_s-1
        # 当x_min小于0时，取0。  x_max小于0时，取0。
        x_min, x_max = int(max(x_min, 0)), int(max(x_max, 0))
        # 当x_min大于w_s时，取值w_s-1。  x_max大于w_s-1时，取w_s-1。
        x_min, x_max = int(min(x_min, w_s - 1)), int(min(x_max, w_s - 1))
        # 当y_min小于0时，取0。  y_max小于0时，取0。
        y_min, y_max = int(max(y_min, 0)), int(max(y_max, 0))
        # 当y_min大于h_s时，取值h_s-1。  y_max大于h_s-1时，取h_s-1。
        y_min, y_max = int(min(y_min, h_s - 1)), int(min(y_max, h_s - 1))
        return Rect(x=x_min, y=y_min, width=(x_max - x_min), height=(y_max - y_min))

    @staticmethod
    def delect_rect_descriptors(rect: Rect, kp: List[cv2.KeyPoint], des: np.ndarray):
        """
        删除rect范围内的特征点与描述符
        """
        tl, br = rect.tl, rect.br

        delect_list = tuple(kp.index(i) for i in kp if tl.x <= i.pt[0] <= br.x and tl.y <= i.pt[1] <= br.y)
        for i in sorted(delect_list, reverse=True):
            kp.pop(i)

        des = np.delete(des, delect_list, axis=0)
        return kp, des

    @staticmethod
    def delect_good_descriptors(good: List[cv2.DMatch], kp: List[cv2.KeyPoint], des: np.ndarray):
        """
        将匹配的特征点与描述符删除
        """
        delect_list = [i.trainIdx for i in good]
        for i in sorted(delect_list, reverse=True):
            kp.pop(i)

        des = np.delete(des, delect_list, axis=0)
        return kp, des

    def get_extractor_parameters(self):
        return self.extractor_parameters
