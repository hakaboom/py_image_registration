"""
python setup.py sdist
twine upload dist/*
"""
# from image_registration import SIFT
from baseImage import IMAGE
# from image_registration.keypoint_matching.sift import SIFT
from image_registration.template_matching import *
import image_registration

match = image_registration.MatchTemplate()
# orb = ORB()

im_source = IMAGE()
im_search = IMAGE('./test/image/star.png')

print(match.find_best_result(im_source='./test/image/test.png', im_search=im_search))
