"""
python setup.py sdist
twine upload dist/*
"""
from image_registration import match_template

tpl = match_template()
tpl.find_templates('test.png', 'star.png')
