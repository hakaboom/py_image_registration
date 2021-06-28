"""
python setup.py sdist
twine upload dist/*
"""
import cv2
from loguru import logger
from baseImage import IMAGE
from collections import OrderedDict
from image_registration import match_template, ORB, CUDA_ORB, RootSIFT, CUDA_SURF, SURF
from image_registration.exceptions import NoEnoughPointsError, CreateExtractorError, BaseError
from image_registration.utils import pprint


CVSTRATEGY = [ORB, RootSIFT]
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    CVSTRATEGY = [match_template, CUDA_ORB, RootSIFT, CUDA_SURF]
CVPARAMS = {
    'ORB': [
        dict(),
        dict(nfeatures=50000, scaleFactor=2, nlevels=2, firstLevel=2),
        dict(nfeatures=50000, scaleFactor=2, nlevels=4, firstLevel=2),
    ]
}


def init_matching_methods():
    MATCH_METHODS = OrderedDict()

    for method in CVSTRATEGY:
        if method.__name__ in CVPARAMS:
            param_config = CVPARAMS[method.__name__]
        else:
            param_config = None

        if isinstance(param_config, list):
            MATCH_METHODS[method.__name__] = []
            for cfg in param_config:
                func = _create_extractor(method, cfg)
                if func:
                    MATCH_METHODS[method.__name__].append(func)
        elif isinstance(param_config, dict):
            func = _create_extractor(method, param_config)
            if func:
                MATCH_METHODS[method.__name__] = func
        else:
            func = _create_extractor(method)
            if func:
                MATCH_METHODS[method.__name__] = func

    return MATCH_METHODS


def _create_extractor(method, config=None):
    try:
        if config:
            func = method(**config)
        else:
            func = method()
    except CreateExtractorError:
        logger.error('create {} config=({}) error', method.__name__, config)
        import traceback
        traceback.print_exc()
        return None
    else:
        return func


MATCH_METHODS = init_matching_methods()

img_source = IMAGE('test.png')
img_search = IMAGE('star.png')


def cv_match(im_source, im_search):
    for method_name, method in MATCH_METHODS.items():
        if isinstance(method, list):
            for func in method:
                result = _try_find_best(func, im_source=im_source, im_search=im_search)
                if result:
                    return result
        else:
            result = _try_find_best(method, im_source=im_source, im_search=im_search)
            if result:
                return result

    return None


def _try_find_best(func, im_source, im_search):
    try:
        match_result = func.find_best(im_source=im_source, im_search=im_search)
    except BaseError as err:
        logger.error('{} \nparam:{}', err, func.get_extractor_parameters())
        return None
    else:
        return match_result


# ret = cv_match(im_source=img_source, im_search=img_search)
# logger.info(ret)
