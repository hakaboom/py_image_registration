"""
python setup.py sdist
twine upload dist/*
"""
import cv2
from baseImage import IMAGE
from image_registration import match_template, ORB, SIFT, RootSIFT

tpl = match_template()
sift = SIFT()
# orb = ORB(scaleFactor=2, nlevels=3, firstLevel=2)
orb = ORB()



im_source = IMAGE('test.png')
im_search = IMAGE('star.png')

# tpl.find_templates(im_source, im_search)
orb.find_best(im_source='star.png', im_search='star.png')

