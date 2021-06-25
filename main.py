"""
python setup.py sdist
twine upload dist/*
"""

from image_registration import match_template, ORB, SIFT, RootSIFT, SURF

# tpl = match_template()
# tpl.find_templates('test.png', 'star.png')
# #
orb = ORB(scaleFactor=2, nlevels=2, firstLevel=1)

orb.find_all(im_source='test.png', im_search='star.png')
