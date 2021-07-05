import cv2

from .keypoint_contrib import RootSIFT, ORB, CUDA_ORB
from .keypoint_matching import match_template

CVSTRATEGY = [match_template, ORB, RootSIFT]
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    CVSTRATEGY = [match_template, CUDA_ORB, RootSIFT]

CVPARAMS = {
    'ORB': [
        dict(nfeatures=60000),
        dict(nfeatures=60000, scaleFactor=2, nlevels=2, firstLevel=1),
        dict(nfeatures=60000, scaleFactor=2, nlevels=4, firstLevel=2),
    ]
}