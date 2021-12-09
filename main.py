"""
python setup.py sdist
twine upload dist/*
"""
import time

from baseImage import Image
from image_registration import CudaMatchTemplate, MatchTemplate


im_source = Image('./test/image/test.png')
im_search = Image('./test/image/test2.png')
im_source.transform_gpu()
im_search.transform_gpu()
# tpl = CudaMatchTemplate()
tpl = MatchTemplate()

tpl.find_best_result(im_source=im_source, im_search=im_search)
start_time = time.time()
for i in range(100):
    tpl.find_best_result(im_source=im_source, im_search=im_search)
print(time.time() - start_time)