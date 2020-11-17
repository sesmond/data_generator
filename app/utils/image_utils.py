#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Title   :TODO
@File    :   image_utils.py    
@Author  : vincent
@Time    : 2020/11/17 5:22 下午
@Version : 1.0 
'''
import matplotlib.pyplot as plt
from PIL import Image


def show(img):
    plt.imshow(img)
    plt.show()


def read(img_path):
    image = Image.open(img_path)
    if image.mode == "L":
        # logger.error("图像[%s]是灰度的，转RGB",img_name)
        image = image.convert("RGB")
    return image
