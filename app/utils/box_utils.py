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
# from shapely.geometry import Polygon, MultiPoint  # 多边形
from sympy.geometry import Polygon, Point


class Box:
    def __init__(self, box):
        box = np.array(box).reshape(-1, 2)
        self.box = box
        box_list = box.tolist()
        p1, p2, p3, p4 = map(Point, box_list)
        # self.poly = Polygon(box).convex_hull
        self.poly = Polygon(p1, p2, p3, p4)
        min_x = min(box[:, 0])
        max_x = max(box[:, 0])
        min_y = min(box[:, 1])
        max_y = max(box[:, 1])
        self.rect = [min_x, min_y, max_x, max_y]

    @staticmethod
    def mat_inter(box1, box2):
        # 判断两个矩形是否相交
        # box=(xA,yA,xB,yB)
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2

        lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
        ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
        sax = abs(x01 - x02)
        sbx = abs(x11 - x12)
        say = abs(y01 - y02)
        sby = abs(y11 - y12)
        if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
            return True
        else:
            return False

    def intersection_area(self, box):
        p_a = self.poly  # sympy.geometry.Polygon
        p_b = box.poly
        check = self.mat_inter(self.rect, box.rect)
        if not check:
            return 0
        x_obj = p_a.intersection(p_b)
        if not x_obj:
            if p_b.encloses_point(p_a.vertices[0]):
                return p_a.area
            if p_a.encloses_point(p_b.vertices[0]):
                return p_b.area
            return 0
        # TODO 有交点，相交面积不算了，直接返回1就可以
        return 1


def intersection_area(box_a, box_b):
    # line1 = [908, 215, 934, 312, 752, 355, 728, 252]  # 四边形四个点坐标的一维数组表示，[x,y,x,y....]
    # a = np.array(line1).reshape(4, 2)  # 四边形二维坐标表示
    # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上

    poly1 = Polygon(box_a).convex_hull
    poly2 = Polygon(box_b).convex_hull
    inter_area = poly1.intersection(poly2).area  # 相交面积
    print(inter_area)
    return inter_area
