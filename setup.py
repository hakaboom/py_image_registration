from setuptools import setup

setup(
    name='py-image-registration',
    version='1.0.11',
    author='hakaboom',
    license="Apache License 2.0",
    author_email='1534225986@qq.com',
    description='image registration by ORB/SIFT/SURF/CUDA_SURF/ORB/CUDA_ORB/matchTemplate',
    url='https://github.com/hakaboom/py_image_registration',
    packages=['image_registration'],
    install_requires=['colorama>=0.4.4',
                      "loguru>=0.5.3",
                      "baseImage==1.0.6",
],
)