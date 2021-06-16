#! usr/bin/python
# -*- coding:utf-8 -*-
""" opencv matchTemplate"""
import cv2
import time
from .utils import generate_result
from baseImage import IMAGE, Rect
from loguru import logger
from typing import Union


class _match_template(object):
    METHOD_NAME = "tpl"

    def __init__(self):
        self.threshold = 0.85

    def find_template(self, im_source, im_search, threshold: Union[int, float] = None, rgb: bool = True):
        """
        模板匹配
        :param im_source: 待匹配图像
        :param im_search: 待匹配模板
        :param threshold: 匹配度 0~1
        :param rgb: 是否判断rgb颜色
        :return: None or Rect
        """
        start = time.time()
        im_source, im_search = self.check_detection_input(im_source, im_search)
        result = self._get_template_result_matrix(im_source, im_search)
        # 找到最佳匹配项
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        h, w = im_search.size
        # 求可信度
        img_crop = im_source.crop_image(Rect(max_loc[0], max_loc[1], w, h))
        confidence = self._get_confidence_from_matrix(img_crop, im_search, max_val=max_val, rgb=rgb)
        # 如果可信度小于threshold,则返回None
        if confidence < (threshold or self.threshold):
            return None
        x, y = max_loc
        rect = Rect(x=x, y=y, width=w, height=h)
        logger.info('[{METHOD_NAME}]{Rect}, confidence={confidence}, time={time:.2f}ms'.format(
            METHOD_NAME=self.METHOD_NAME, confidence=confidence, Rect=rect, time=(time.time() - start) * 1000))
        return generate_result(rect, confidence)

    def find_templates(self, im_source, im_search, threshold: Union[int, float] = None,
                       max_count: int = 10,
                       rgb: bool = True):
        """
        模板匹配
        :param im_source: 待匹配图像
        :param im_search: 待匹配模板
        :param threshold: 匹配度
        :param max_count: 最多匹配数量
        :param rgb: 是否判断rgb颜色
        :return: None or Rect
        """
        start = time.time()
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
        if result:
            print('[{METHOD_NAME}s] find counts:{counts}, time={time:.2f}ms{result}'.format(
                METHOD_NAME=self.METHOD_NAME,
                counts=len(result), time=(time.time() - start) * 1000,
                result=''.join(['\n\t{}, confidence={}'.format(x['rect'], x['confidence'])for x in result])))
        return result if result else None

    @staticmethod
    def _get_template_result_matrix(im_source, im_search):
        """求取模板匹配的结果矩阵."""
        s_gray, i_gray = im_search.rgb_2_gray(), im_source.rgb_2_gray()
        return cv2.matchTemplate(i_gray, s_gray, cv2.TM_CCOEFF_NORMED)

    @staticmethod
    def check_detection_input(im_source, im_search):
        return IMAGE(im_source), IMAGE(im_search)

    @staticmethod
    def cal_rgb_confidence(img_src_rgb, img_sch_rgb):
        img_src_rgb, img_sch_rgb = img_src_rgb.imread(), img_sch_rgb.imread()
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
    def _get_confidence_from_matrix(img_crop, im_search, max_val, rgb):
        """根据结果矩阵求出confidence."""
        # 求取可信度:
        if rgb:
            # 如果有颜色校验,对目标区域进行BGR三通道校验:
            confidence = _match_template.cal_rgb_confidence(img_crop, im_search)
        else:
            confidence = max_val
        return confidence


class _cuda_match_template(_match_template):
    METHOD_NAME = "cuda_tpl"

    def __init__(self):
        super(_cuda_match_template, self).__init__()
        self.matcher = cv2.cuda.createTemplateMatching(cv2.CV_8U, cv2.TM_CCOEFF_NORMED)

    @staticmethod
    def check_detection_input(im_source, im_search):
        im_source, im_search = IMAGE(im_source), IMAGE(im_search)
        im_source.transform_gpu()
        im_search.transform_gpu()
        return im_source, im_search

    def _get_template_result_matrix(self, im_source, im_search):
        """求取模板匹配的结果矩阵."""
        s_gray, i_gray = im_search.rgb_2_gray(), im_source.rgb_2_gray()
        res = self.matcher.match(i_gray, s_gray)
        return res.download()

    def cuda_cal_rgb_confidence(self, img_src_rgb, img_sch_rgb):
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


if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    class match_template(_cuda_match_template):
        pass
else:
    class match_template(_match_template):
        pass
