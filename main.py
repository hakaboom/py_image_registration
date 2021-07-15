"""
python setup.py sdist
twine upload dist/*
"""

from baseImage import IMAGE
from image_registration.findit import Findit

match = Findit()
# orb = ORB()

im_source = IMAGE()
im_search = IMAGE('./test/image/star.png')

print(match.find_best_result(im_source='./test/image/test.png', im_search=im_search))
