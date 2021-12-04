# -*- coding: utf-8 -*-
import cv2
import numpy as np
from typing import Union, Tuple
from baseImage import Image

from image_registration.template_matching.matchTemplate import MatchTemplate
from image_registration.exceptions import NoModuleError


class CudaMatchTemplate(MatchTemplate):
    METHOD_NAME = "cuda_tpl"

    def __init__(self, threshold: Union[int, float] = 0.8, rgb: bool = True):
        """
        初始化
        :param threshold: 识别阈值(0~1)
        :param rgb: 是否使用rgb通道进行校验
        :return: None
        """
        super(CudaMatchTemplate, self).__init__(threshold, rgb)
        try:
            self.matcher = cv2.cuda.createTemplateMatching(cv2.CV_8U, cv2.TM_CCOEFF_NORMED)
        except AttributeError:
            raise NoModuleError('create CUDA TemplateMatching Error')

    @staticmethod
    def check_detection_input(im_source: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes],
                              im_search: Union[Image, str, np.ndarray, cv2.cuda_GpuMat, bytes]) -> Tuple[Image, Image]:
        if not isinstance(im_source, Image):
            im_source = Image(im_source)
        if not isinstance(im_search, Image):
            im_search = Image(im_search)

        im_source.transform_gpu()
        im_search.transform_gpu()
        return im_source, im_search

    def _get_template_result_matrix(self, im_source: Image, im_search: Image):
        """求取模板匹配的结果矩阵."""
        s_gray, i_gray = im_search.rgb_2_gray(), im_source.rgb_2_gray()
        res = self.matcher.match(i_gray, s_gray)
        return res.download()

    def cuda_cal_rgb_confidence(self, img_src_rgb, img_sch_rgb):
        """
        计算两张图片图片rgb三通道的置信度
        :param img_src_rgb: 待匹配图像
        :param img_sch_rgb: 图片模板
        :return: 最小置信度
        """
        img_src_rgb, img_sch_rgb = img_src_rgb.download(), img_sch_rgb.download()
        img_sch_rgb = cv2.cuda.copyMakeBorder(img_sch_rgb, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
        # 转HSV强化颜色的影响
        img_src_rgb = cv2.cuda.cvtColor(img_src_rgb, cv2.COLOR_BGR2HSV)
        img_sch_rgb = cv2.cuda.cvtColor(img_sch_rgb, cv2.COLOR_BGR2HSV)
        src_bgr, sch_bgr = cv2.cuda.split(img_src_rgb), cv2.cuda.split(img_sch_rgb)
        # 计算BGR三通道的confidence，存入bgr_confidence:
        bgr_confidence = [0, 0, 0]
        for i in range(3):
            res_temp = self.matcher.match(sch_bgr[i], src_bgr[i]).download()
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_temp)
            bgr_confidence[i] = max_val
        return min(bgr_confidence)

    def _get_confidence_from_matrix(self, img_crop, im_search, max_val, rgb):
        """根据结果矩阵求出confidence."""
        # 求取可信度:
        if rgb:
            # 如果有颜色校验,对目标区域进行BGR三通道校验:
            confidence = self.cuda_cal_rgb_confidence(img_crop, im_search)
        else:
            confidence = max_val
        return confidence
