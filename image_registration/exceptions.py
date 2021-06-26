class BaseError(Exception):
    """ There was an exception that occurred while handling BaseImage"""

    def __init__(self, message="", *args, **kwargs):
        self.message = message

    def __repr__(self):
        return repr(self.message)


class NoModuleError(BaseError):
    """ Missing dependent module """


class ExtractorError(BaseError):
    """ An error occurred while create Extractor """


class NoEnoughPointsError(BaseError):
    """ detect not enough feature points in input images"""


class CudaSuftInputImageError(BaseError):
    """ The image size does not conform to CUDA standard  """
    # https://stackoverflow.com/questions/42492060/surf-cuda-error-while-computing-descriptors-and-keypoints
    # https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/surf.cuda.cpp#L151


class CudaOrbDetectorError(BaseError):
    """ An CvError when orb detector error occurred """
