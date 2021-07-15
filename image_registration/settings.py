import cv2
# TODO: 修改接口函数
from image_registration import (ORB, RootSIFT, SIFT, SURF, KAZE, BRIEF, MatchTemplate,
                                CUDA_SURF, CUDA_ORB, CudaMatchTemplate)

CVSTRATEGY = [MatchTemplate, ORB, RootSIFT]
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    CVSTRATEGY = [CudaMatchTemplate, CUDA_ORB, RootSIFT]

CVPARAMS = {
    'ORB': [
        dict(nfeatures=60000),
        dict(nfeatures=60000, scaleFactor=2, nlevels=2, firstLevel=1),
        dict(nfeatures=60000, scaleFactor=2, nlevels=4, firstLevel=2),
    ]
}