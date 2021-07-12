"""
python setup.py sdist
twine upload dist/*
"""
from image_registration import SIFT
from baseImage import IMAGE

sift = SIFT()

im_source = IMAGE('./test/image/test.png')
im_search = IMAGE('./test/image/star.png')

sift.find_best(im_source=im_source, im_search=im_search)
