#! usr/bin/python
# -*- coding:utf-8 -*-
""" opencv matchTemplate"""
import cv2
import numpy as np

from image_registration.exceptions import (MatchResultError)
from image_registration.utils import generate_result
from baseImage import Image, Rect
from typing import Union, Tuple


class MatchTemplate(object):
    METHOD_NAME = "tpl"

    def __init__(self, threshold: Union[int, float] = 0.8, rgb: bool = True):
        """
        初始化
        :param threshold: 识别阈值(0~1)
        :param rgb: 是否使用rgb通道进行校验
        :return: None
        """
        self.threshold = threshold
        self.rgb = rgb

    def find_best_result(self, im_source: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                         im_search: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                         threshold: Union[int, float] = None, rgb: bool = True):
        """
        模板匹配
        :param im_source: 待匹配图像
        :param im_search: 图片模板
        :param threshold: 识别阈值(0~1)
        :param rgb: 是否使用rgb通道进行校验
        :return: None or Rect
        """
        im_source, im_search = self.check_detection_input(im_source, im_search)
        result = self._get_template_result_matrix(im_source, im_search)
        # 找到最佳匹配项
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        h, w = im_search.size
        # 求可信度
        crop_rect = Rect(max_loc[0], max_loc[1], w, h)
        confidence = self.cal_confidence(im_source, im_search, crop_rect, max_val, rgb)
        # 如果可信度小于threshold,则返回None

        if confidence < (threshold or self.threshold):
            return None
        x, y = max_loc
        rect = Rect(x=x, y=y, width=w, height=h)
        return generate_result(rect, confidence)

    def find_all_results(self, im_source: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                         im_search: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                         threshold: Union[int, float] = None, max_count: int = 10, rgb: bool = True):
        """
        模板匹配
        :param im_source: 待匹配图像
        :param im_search: 图片模板
        :param threshold: 识别阈值(0~1)
        :param max_count: 最多匹配数量
        :param rgb: 是否使用rgb通道进行校验
        :return: None or Rect
        """
        im_source, im_search = self.check_detection_input(im_source, im_search)
        # 模板匹配取得矩阵
        res = self._get_template_result_matrix(im_source, im_search)
        result = []
        h, w = im_search.size
        while True:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            img_crop = im_source.crop_image(Rect(max_loc[0], max_loc[1], w, h))
            confidence = self._get_confidence_from_matrix(img_crop, im_search, max_val=max_val, rgb=rgb)
            x, y = max_loc
            rect = Rect(x, y, w, h)
            if (confidence < (threshold or self.threshold)) or len(result) >= max_count:
                break
            result.append(generate_result(rect, confidence))
            cv2.rectangle(res, (int(max_loc[0] - w / 2), int(max_loc[1] - h / 2)),
                          (int(max_loc[0] + w / 2), int(max_loc[1] + h / 2)), (0, 0, 0), -1)
        return result if result else None

    @staticmethod
    def _get_template_result_matrix(im_source: Image, im_search: Image) -> np.ndarray:
        """求取模板匹配的结果矩阵."""
        s_gray, i_gray = im_search.rgb_2_gray(), im_source.rgb_2_gray()
        return cv2.matchTemplate(i_gray, s_gray, cv2.TM_CCOEFF_NORMED)

    @staticmethod
    def check_detection_input(im_source: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                              im_search: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes]) -> Tuple[Image, Image]:
        """
        检测输入的图像数据是否正确
        :param im_source: 待匹配图像
        :param im_search: 图片模板
        :return: im_source, im_search
        """
        if not isinstance(im_source, Image):
            im_source = Image(im_source)
        if not isinstance(im_search, Image):
            im_search = Image(im_search)

        im_source.transform_cpu()
        im_search.transform_cpu()
        return im_source, im_search

    @staticmethod
    def cal_rgb_confidence(im_source: Image, im_search: Image):
        """
        计算两张图片图片rgb三通道的置信度
        :param im_source: 待匹配图像
        :param im_search: 图片模板
        :return: 最小置信度
        """
        img_src_rgb, img_sch_rgb = im_source.imread(), im_search.imread()
        img_sch_rgb = cv2.copyMakeBorder(img_sch_rgb, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
        # 转HSV强化颜色的影响
        img_src_rgb = cv2.cvtColor(img_src_rgb, cv2.COLOR_BGR2HSV)
        img_sch_rgb = cv2.cvtColor(img_sch_rgb, cv2.COLOR_BGR2HSV)
        src_bgr, sch_bgr = cv2.split(img_src_rgb), cv2.split(img_sch_rgb)
        # 计算BGR三通道的confidence，存入bgr_confidence:
        bgr_confidence = [0, 0, 0]
        for i in range(3):
            res_temp = cv2.matchTemplate(src_bgr[i], sch_bgr[i], cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_temp)
            bgr_confidence[i] = max_val
        return min(bgr_confidence)

    @staticmethod
    def cal_ccoeff_confidence(im_source: Image, im_search: Image):
        """
        使用CCOEFF方法模板匹配图像
        """
        img_src_gray, img_sch_gray = im_source.rgb_2_gray(), im_search.rgb_2_gray()
        # 扩展置信度计算区域
        img_sch_gray = cv2.copyMakeBorder(img_sch_gray, 10, 10, 10, 10, cv2.BORDER_REPLICATE)

        res = cv2.matchTemplate(img_src_gray, img_sch_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        return max_val

    def cal_confidence(self, im_source: Image, im_search: Image, crop_rect: Rect, max_val, rgb) -> Union[int, float]:
        """
        将截图和识别结果缩放到大小一致,并计算可信度
        :param im_source: 待匹配图像
        :param im_search: 图片模板
        :param crop_rect: 需要在im_source截取的区域
        :param rgb: 是否使用rgb通道进行校验
        :param max_val: matchTemplate得到的最大值
        :raise MatchResultError: crop_rect范围超出了im_source边界
        :return: 返回可信度(0~1)
        """
        try:
            target_img = im_source.crop_image(crop_rect)
        except OverflowError:
            raise MatchResultError(f"Target area({crop_rect}) out of screen{im_source.size}")

        confidence = self._get_confidence_from_matrix(target_img, im_search, max_val, rgb)
        return confidence

    @staticmethod
    def _get_confidence_from_matrix(img_crop, im_search, max_val, rgb):
        """根据结果矩阵求出confidence."""
        # 求取可信度:
        if rgb:
            # 如果有颜色校验,对目标区域进行BGR三通道校验:
            confidence = MatchTemplate.cal_rgb_confidence(img_crop, im_search)
        else:
            confidence = max_val
        return confidence
