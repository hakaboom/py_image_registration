import cv2
from image_registration import (ORB, RootSIFT, SIFT, SURF, KAZE, BRIEF, MatchTemplate,
                                CUDA_SURF, CUDA_ORB, CudaMatchTemplate)

CVSTRATEGY = [MatchTemplate, ORB, RootSIFT]
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    CVSTRATEGY = [CudaMatchTemplate, CUDA_ORB, RootSIFT]

CVPARAMS = {
    'CUDA_ORB': [
        dict(nfeatures=60000, scaleFactor=2, nlevels=1, firstLevel=1),
    ]
}