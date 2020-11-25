#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Title   :TODO
@File    :   box_utils.py    
@Author  : vincent
@Time    : 2020/11/25 下午6:34
@Version : 1.0 
'''
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形

class Box:
    def __init__(self, box):
        box = np.array(box).reshape(-1, 2)
        self.box = box
        self.poly = Polygon(box).convex_hull

    def intersection_area(self, box):
        inter_area = self.poly.intersection(box.poly).area  # 相交面积
        return inter_area


def intersection_area(box_a, box_b):
    # line1 = [908, 215, 934, 312, 752, 355, 728, 252]  # 四边形四个点坐标的一维数组表示，[x,y,x,y....]
    # a = np.array(line1).reshape(4, 2)  # 四边形二维坐标表示
    # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上

    poly1 = Polygon(box_a).convex_hull
    poly2 = Polygon(box_b).convex_hull
    inter_area = poly1.intersection(poly2).area  # 相交面积
    print(inter_area)
    return inter_area
