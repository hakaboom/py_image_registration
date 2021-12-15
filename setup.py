from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='py-image-registration',
    version='1.0.16',
    author='hakaboom',
    license="Apache License 2.0",
    author_email='1534225986@qq.com',
    description='image registration by ORB/SIFT/SURF/CUDA_SURF/ORB/CUDA_ORB/matchTemplate',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hakaboom/py_image_registration',
    packages=find_packages(),
    include_package_data=True,
    install_requires=["loguru>=0.5.3",
                      "baseImage>=1.0.9"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)