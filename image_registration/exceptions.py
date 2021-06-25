
class SurfCudaError(Exception):
    def __init__(self, image):
        self.image = image

    def __str__(self):
        return '{width}x{height}'.format(width=self.image.size[1], height=self.image.size[0])


class CvError(SurfCudaError):
    def __init__(self, image):
        self.image = image


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



