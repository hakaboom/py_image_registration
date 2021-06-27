#! usr/bin/python
# -*- coding:utf-8 -*-
import time
import cv2
import numpy
import numpy as np
from .utils import generate_result, match_time_debug, print_all_result, print_best_result
from .exceptions import NoEnoughPointsError, HomographyError, MatchResultError, PerspectiveTransformError
from .match_template import match_template
from baseImage import IMAGE, Rect, Point, Size
from loguru import logger
from typing import Tuple, List, Union


class KeypointMatch(object):
    FLANN_INDEX_KDTREE = 0
    FILTER_RATIO = 0.59
    METHOD_NAME = 'KeypointMatch'
    template = match_template()

    def __init__(self, threshold: Union[int, float] = 0.8):
        self.threshold = threshold
        self.matcher = self.create_matcher()
        self.detector = cv2.KAZE_create()

    def create_matcher(self) -> cv2.FlannBasedMatcher:
        index_params = {'algorithm': self.FLANN_INDEX_KDTREE, 'tree': 5}
        # 指定递归遍历的次数. 值越高结果越准确，但是消耗的时间也越多
        search_params = {'checks': 50}
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        return matcher

    @print_best_result
    def find_best(self, im_source, im_search, threshold: Union[int, float] = None):
        """在im_source中找到最符合im_search的范围"""
        threshold = threshold is None and self.threshold or threshold
        im_source, im_search = self.check_detection_input(im_source, im_search)
        if not im_source or not im_search:
            return None
        # 第一步: 获取特征点集
        kp_sch, des_sch = self.get_keypoints_and_descriptors(image=im_search.rgb_2_gray())
        kp_src, des_src = self.get_keypoints_and_descriptors(image=im_source.rgb_2_gray())
        # 第二步: 在特征点集中匹配最接近的范围
        rect, matches, good = self.get_rect_from_good_matches(im_source, im_search, kp_sch, des_sch, kp_src, des_src)
        if not rect:
            return None
        # 第三步, 从匹配图片截取矩阵范围,并缩放到模板大小,进行模板匹配求出相似度
        confidence = self._cal_confidence(im_source=im_source, im_search=im_search, rect=rect)
        best_match = generate_result(rect=rect, confi=confidence)
        return best_match if confidence > threshold else None

    @print_all_result
    def find_all(self, im_source, im_search, threshold: Union[int, float] = None):
        threshold = threshold is None and self.threshold or threshold
        im_source, im_search = self.check_detection_input(im_source, im_search)
        if not im_source or not im_search:
            return None
        result = []
        # 第一步: 获取特征点集
        kp_sch, des_sch = self.get_keypoints_and_descriptors(image=im_search.rgb_2_gray())
        kp_src, des_src = self.get_keypoints_and_descriptors(image=im_source.rgb_2_gray())

        while len(kp_src) > 2 or len(kp_sch) > 2:
            rect, matches, good = self.get_rect_from_good_matches(im_source, im_search,
                                                                  kp_sch, des_sch,
                                                                  kp_src, des_src)
            if not rect:
                break

            confidence = self._cal_confidence(im_source=im_source, im_search=im_search, rect=rect)
            if confidence > threshold:
                result.append(generate_result(rect, confidence))

                kp_src, des_src = self.delect_good_descriptors(good, kp_src, des_src)
                # 无特征点时, 结束函数
                if len(kp_src) < 2 or len(kp_sch) < 2:
                    break

                kp_src, des_src = self.delect_rect_descriptors(rect, kp_src, des_src)
            else:
                # 未找到其他匹配区域,退出寻找
                break
        return result

    def _cal_confidence(self, im_source, im_search, rect):
        """ 将截图和识别结果缩放到大小一致,并计算可信度 """
        try:
            target_img = im_source.crop_image(rect)
        except OverflowError:
            raise MatchResultError("Target area({}) out of screen{}".format(rect, im_source.size))

        h, w = im_search.size
        target_img.resize(w, h)

        confidence = self.template.cal_rgb_confidence(img_src_rgb=im_search, img_sch_rgb=target_img)
        confidence = (1 + confidence) / 2
        return confidence

    @staticmethod
    def check_detection_input(im_source, im_search) -> Tuple[IMAGE, IMAGE]:
        if not isinstance(im_source, IMAGE):
            im_source = IMAGE(im_source)
        if not isinstance(im_search, IMAGE):
            im_search = IMAGE(im_search)

        im_source.transform_cpu()
        im_search.transform_cpu()
        return im_source, im_search

    def extract_good_points(self, im_source, im_search, kp_sch, kp_src, good):
        if len(good) in [0, 1]:
            # origin_result = self._handle_one_good_points(im_source, im_search, kp_src, kp_sch, good)
            return None
        elif len(good) in [2, 3]:
            if len(good) == 2:
                # 匹配点对为2，根据点对求出目标区域，据此算出可信度：
                origin_result = self._handle_two_good_points(im_source, im_search, kp_src, kp_sch, good)
            else:
                origin_result = self._handle_three_good_points(im_source, im_search, kp_sch, kp_src, good)
            if isinstance(origin_result, dict):
                return None
            else:
                return origin_result
        else:
            # 匹配点大于4,使用单矩阵映射求出目标区域
            return self._many_good_pts(im_source, im_search, kp_sch, kp_src, good)

    @match_time_debug
    def get_rect_from_good_matches(self, im_source, im_search, kp_sch, des_sch, kp_src, des_src):
        matches = self.match_keypoints(des_sch=des_sch, des_src=des_src)
        good = self.get_good_in_matches(matches)
        rect = self.extract_good_points(im_source, im_search, kp_sch, kp_src, good)
        return rect, matches, good

    def get_keypoints_and_descriptors(self, image: numpy.ndarray) -> Tuple[List[cv2.KeyPoint], numpy.ndarray]:
        keypoints, descriptors = self.detector.detectAndCompute(image, None)

        if len(keypoints) < 2:
            raise NoEnoughPointsError
        return keypoints, descriptors

    def match_keypoints(self, des_sch: numpy.ndarray, des_src: numpy.ndarray) -> List[List[cv2.DMatch]]:
        """Match descriptors (特征值匹配)."""
        # 匹配两个图片中的特征点集，k=2表示每个特征点取出2个最匹配的对应点:
        matches = self.matcher.knnMatch(des_sch, des_src, 2)
        return matches

    def get_good_in_matches(self, matches) -> List[cv2.DMatch]:
        good = []
        for m, n in matches:
            if m.distance < self.FILTER_RATIO * n.distance:
                good.append(m)
        return good

    @staticmethod
    def _find_homography(sch_pts, src_pts):
        """多组特征点对时，求取单向性矩阵."""
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

    def _many_good_pts(self, im_source, im_search, kp_sch, kp_src, good) -> Rect:
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

    def _handle_one_good_points(self, im_source, im_search, kp_src, kp_sch, good):
        """匹配中只有一对匹配的特征点对的情况."""
        """此方法当前废弃"""
        raise NotImplementedError
        # 取出该点在图中的位置
        # sch_point = Point(int(kp_sch[0].pt[0]), int(kp_sch[0].pt[1]))
        # src_point = Point(int(kp_src[good[0].trainIdx].pt[0]), int(kp_src[good[0].trainIdx].pt[1]))
        # # 求出模板原点在匹配图像上的坐标
        # offset_point = src_point - sch_point
        # rect = Rect.create_by_point_size(offset_point, Size(im_search.shape[1], im_search.shape[0]))
        # logger.debug('rect={},sch={}, src={}, offset={}', rect, sch_point, src_point, offset_point)
        # return rect

    def _handle_two_good_points(self, im_source, im_search, kp_src, kp_sch, good):
        """处理两对特征点的情况."""
        pts_sch1 = int(kp_sch[good[0].queryIdx].pt[0]), int(kp_sch[good[0].queryIdx].pt[1])
        pts_sch2 = int(kp_sch[good[1].queryIdx].pt[0]), int(kp_sch[good[1].queryIdx].pt[1])
        pts_src1 = int(kp_src[good[0].trainIdx].pt[0]), int(kp_src[good[0].trainIdx].pt[1])
        pts_src2 = int(kp_src[good[1].trainIdx].pt[0]), int(kp_src[good[1].trainIdx].pt[1])

        return self._two_good_points(pts_sch1, pts_sch2, pts_src1, pts_src2, im_search, im_source)

    def _handle_three_good_points(self, im_source, im_search, kp_sch, kp_src, good):
        """处理三对特征点的情况."""
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

    @staticmethod
    def delect_rect_descriptors(rect, kp, des):
        tl, br = rect.tl, rect.br
        kp = kp.copy()
        des = des.copy()

        delect_list = tuple(kp.index(i) for i in kp if tl.x <= i.pt[0] <= br.x and tl.y <= i.pt[1] <= br.y)
        for i in sorted(delect_list, reverse=True):
            kp.pop(i)

        des = numpy.delete(des, delect_list, axis=0)
        return kp, des

    @staticmethod
    def delect_good_descriptors(good, kp, des):
        kp = kp.copy()
        des = des.copy()

        delect_list = [i.trainIdx for i in good]
        for i in sorted(delect_list, reverse=True):
            kp.pop(i)

        des = numpy.delete(des, delect_list, axis=0)
        return kp, des
