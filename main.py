"""
python setup.py sdist
twine upload dist/*
"""
# from image_registration import SIFT
from baseImage import IMAGE
from image_registration.keypoint_matching.sift import SIFT

sift = SIFT()
# orb = ORB()

im_source = IMAGE('./test/image/test.png')
im_search = IMAGE('./test/image/test.png')

print(sift.find_best_result(im_source=im_source, im_search=im_search))
