class SurfCudaError(Exception):
    def __init__(self, image):
        self.image = image

    def __str__(self):
        return '{width}x{height}'.format(width=self.image.size[1], height=self.image.size[0])


class CvError(SurfCudaError):
    def __init__(self, image):
        self.image = image