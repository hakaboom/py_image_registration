from setuptools import setup

setup(
    name='image_registration',
    version='1.0.0',
    author='hakaboom',
    author_email='1534225986@qq.com',
    description='image registration by ORB/SIFT/SURF/CUDA_SURF/ORB/CUDA_ORB/matchTemplate',
    url='https://github.com/hakaboom/py_image_registration',
    packages=['baseImage'],
    install_requires=['colorama>=0.4.4',
                      "loguru>=0.5.3",
],
)