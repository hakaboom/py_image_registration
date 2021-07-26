# py_image_registration
[![GitHub issues](https://img.shields.io/github/issues/hakaboom/py_image_registration)](https://github.com/hakaboom/py_image_registration/issues)
[![GitHub forks](https://img.shields.io/github/forks/hakaboom/py_image_registration)](https://github.com/hakaboom/py_image_registration/network)
[![GitHub stars](https://img.shields.io/github/stars/hakaboom/py_image_registration)](https://github.com/hakaboom/py_image_registration/stargazers)
[![GitHub license](https://img.shields.io/github/license/hakaboom/py_image_registration?style=plastic)](https://github.com/hakaboom/py_image_registration/blob/master/LICENSE)


Image registration algorithm. Includes SIFT, ORB, SURF, AKAZE, BRIEF, matchTemplate

同时包含cuda加速(ORB, SURF, matchTemplate)


## Requirements
- 开发时使用的是python 3.8
- opencv需要自己安装或自行编译,读取的到cv2模块就行

## Installation
pip3 install py_image_registration

## Example

1. **create**

```Python
from image_registration import ORB, SIFT, RootSIFT, SURF, BRIEF, AKAZE, CUDA_SURF, CUDA_ORB, match_template

orb = ORB()
sift = SIFT()
# Other
# orb = ORB(nfeatures=1000, nlevels=9)
# 提取器的参数可以根据opencv文档调整
```


2. **MatchTemplate**

模板匹配
```Python
from image_registration import match_template
from baseImage import IMAGE, Rect

im_source = IMAGE('test.png')
im_search = IMAGE('star.png')

tpl = match_template()
result = tpl.find_best(im_source=im_source, im_search=im_search)
# expect output
# {
#  'rect': Rect,  # 返回一个baseImage.Rect类的识别范围
#  'confidence': 0.99 # 返回识别结果的置信度
# }
# Other
# tpl.find_best(im_source=im_source, im_search=im_search, threshold=0.8, rgb=False)
# threshold: 匹配度 0~1
# rgb: 是否判断rgb颜色


tpl.find_all(im_source=im_source, im_search=im_search)
# expect output
# {
#  {
#     'rect': Rect,  # 返回一个baseImage.Rect类的识别范围
#     'confidence': 0.99 # 返回识别结果的置信度
#  },
#  {
#     'rect': Rect
#     'confidence': 0.95
#  },
#  ...
# }
# Other
# tpl.find_all(im_source=im_source, im_search=im_search, threshold=0.8, max_count=20, rgb=False)
# threshold: 匹配度 0~1
# max_count: 最多匹配数量
# rgb: 是否判断rgb颜色
```

3. **keypoint detector and descriptor extractor**

基于特征点的匹配

```Python
from image_registration import ORB, SIFT, RootSIFT, SURF, BRIEF, AKAZE, CUDA_SURF, CUDA_ORB

orb = ORB()
sift = SIFT()

im_source = IMAGE('test.png')
im_search = IMAGE('star.png')

orb.find_best(im_source=im_source, im_search=im_search)
orb.find_all(im_source=im_source, im_search=im_search)
# 返回结果MatchTemplate
```

## Exceptions

1. **NoModuleError** 

模块未找到, 检查opencv库是否安装正确

2. **CreateExtractorError**

创建特征提取器失败, 检查传入参数以及opencv库是否安装正确

3. **NoEnoughPointsError**

当特征提取器提取特殊数量少于2时弹出异常

4. **CudaSurfInputImageError**

图像大小不符合cuda_surf的要求时弹出异常[opencv_surf.cuda](https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/surf.cuda.cpp#L151)

5. **CudaOrbDetectorError**


提取特征时出现的错误,需要调整orb的初始参数(scaleFactor, nlevels, firstLevel)

6. **HomographyError**


An error occurred while findHomography

7. **MatchResultError**


An error occurred while result out of screen

8. **PerspectiveTransformError**


An error occurred while perspectiveTransform
