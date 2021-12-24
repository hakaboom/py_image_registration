"""
python setup.py sdist
twine upload dist/*
"""
import cv2

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("检测到cuda环境")