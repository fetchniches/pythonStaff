# -*- coding: utf-8 -*-
from typing import Any, List, NewType, Optional, Sequence, Union
import numpy as np
from math import ceil, sqrt
import cv2 as cv
import matplotlib.pyplot as plt
from typing import Union, Optional
import math 
import utils.globalvars as globalv

Array = NewType('Array', np.ndarray)

def stdData(original_data):
    "将数据标准化"
    std_data = []
    for i in range(int(original_data.shape[0]/2)):
        std_data.append([original_data[2*i], original_data[2*i+1]])
    std_data = np.asarray(std_data, dtype=np.float64)

    return std_data


def pointsFilter(array, max_distance, min_distance, min_angle=-45, max_angle=45):
    "在极坐标系下过滤点"
    target_points = []
    for angle, distance in array:
        angle = angle-360 if angle > 180 else angle
        if max_angle > angle > min_angle and max_distance > distance > min_distance:
            target_points.append([angle, distance])
    target_points = np.asarray(target_points, dtype=np.float64)

    return target_points

def pointsSelector(points: np.ndarray, target_angle: float, error_threshold: float, radar_resolution: float = 0.25):
    """
    通过单个已知的角度，获取圆的其他相关点

    参数
    -----
    points: ndarray
        雷达接收的数据，极坐标下
    target_angle: float
        目标角度
    error_threshold: float, 0 < error_threshold < 1
        选取相邻两个极坐标上的点，其长度分别为l1, l2，若 (l1-l2)/l1 < error_threshold 则认为两点都属于同一个圆
    radar_resolution: float
        雷达的分辨率
    
    返回
    -----
    target_points: ndarray
        目标圆的相关点，target_points.shape = (n, 2)，极坐标形式
    当未找到与指定角度相近的点时, 返回None
    
    示例
    -----
    >>> c_tplt = CircleTemplate(...)
    >>> target = pointsSelector(data, 34.21, 0.05)
    >>> polarToCart(target)
    >>> c_tplt(target)
    """
    # 修正角度
    target_angle = round(target_angle/radar_resolution) * radar_resolution
    # 遍历雷达数据
    for index, point in enumerate(points):
        angle = point[0]
        # 找到目标数据
        if angle == target_angle:
            left = True
            right = True
            left_idx = right_idx = index
            # 寻找就近的有效数据的点
            target_points = []
            while True:
                if points[right_idx, 1] != 0:
                    target_points.append(points[right_idx])
                    index = right_idx
                    break
                else:
                    right_idx += 1
                if points[left_idx, 1] != 0:
                    target_points.append(points[left_idx])
                    index = left_idx
                    break
                else:
                    left_idx -= 1
            # 遍历周围点
            left_idx = right_idx = index
            while left or right:
                if right and points[right_idx, 1] != 0 \
                and abs(points[right_idx, 1] - points[right_idx + 1, 1]) / points[right_idx, 1] < error_threshold:
                    print('right +1')
                    target_points.append(points[right_idx + 1])
                    right_idx += 1
                else:
                    print('right end')
                    right = False
                if left and points[left_idx, 1] != 0 \
                and abs(points[left_idx, 1] - points[left_idx - 1, 1]) / points[left_idx, 1] < error_threshold:
                    print('left +1')
                    target_points.append(points[left_idx - 1])
                    left_idx -= 1
                else:
                    print('left end')
                    left = False
            return np.asarray(target_points, dtype=np.float32)
    globalv.get_var('rcmsg').info('未从雷达数据中找到与指定角度{:.2f}相近的点'.format(target_angle))
    return None



def polarToCart(polar_points):
    "极坐标转直角坐标，polar_points为ndarray类型"
    polar_points[:, 0], polar_points[:, 1] = polar_points[:, 1] * np.cos(
        polar_points[:, 0]*np.pi/180), polar_points[:, 1] * np.sin(polar_points[:, 0]*np.pi/180)


def centralize(array):
    "样本中心化，返回中心化前的均值"
    x_mean = array[:, 0].mean()
    y_mean = array[:, 1].mean()
    array[:, 0] -= x_mean
    array[:, 1] -= y_mean

    return x_mean, y_mean

class DynamicPlot(object):
    "绘制动态图"
    def __init__(self, clf: bool = True, pause: float = 0.01):
        super().__init__()
        plt.ion()
        # 开启交互式绘制
        self.clf = clf
        self.pause = pause
    
    def update(self, *args, **kargs):
        "更新画布"
        plt.scatter(*args, **kargs)

    def show(self):
        "显示画布"
        plt.pause(self.pause)
        if self.clf:
            plt.clf()

def drawCircle(centers, radius):
    for center in centers:
        circle = plt.Circle((center[1], center[0]), radius, fill=False)
        plt.gcf().gca().add_artist(circle)


def showMatRes(points, poi, centers, dp, graph=False):
    if graph:
        plt.title('Matching Results')
        dp.update(points[:, 0], points[:, 1], s=0.7)
        dp.update(poi[:, 0], poi[:, 1], s=0.7)
        if centers is not None:
            ...
            # drawCircle(centers, radius=152.5)
        dp.show()


    dis = []
    if centers is not None:
        rcmsg = globalv.get_var('rcmsg')
        for point in centers:
            dis.append((sqrt(point[0]**2+point[1]**2), np.arctan(point[1]/point[0])*(180/np.pi)))
        rcmsg.info('检测到{}个目标 : '.format(len(dis)))
        for i in range(len(dis)):
            rcmsg.info('\t距离: {:.2f}m  角度: {:.2f}°'.format(dis[i][0]/1e3, dis[i][1]))




class CircleTemplate:
    def __init__(self, radius: float, grid_len: int, dilate: float = 0.1):
        """创建圆匹配模板
        
        - 参数:\n
            radius - 待拟合圆半径\n
            grid_len - 网格的长度，越小速度越慢，精度越高\n
            dilate - 匹配半径的膨胀率\n

        """
        self.radius = radius
        self.grid_len = grid_len
        # 添加边界，以增大对直线型点的惩罚
        self.kernel = -1 * np.ones(
            (round((1+dilate)*radius/grid_len+ radius/grid_len), round(2*(1+dilate)*radius/grid_len+radius/grid_len)))
        # 1-dilate倍率的半径长度
        small_radius_on_grid = ceil((1-dilate)*radius/grid_len)
        # 循环中行的边界
        row_bound = ceil(self.kernel.shape[0])
        # 循环中列的边界(取半)
        col_bound = ceil(self.kernel.shape[1])
        # 1+dilate与1-dilate对应倍率的半径
        max_radius_2, min_radius_2 = round(
            ((1+dilate)*radius/grid_len)**2), round(((1-dilate)*radius/grid_len)**2)
        # 尝试搜索算法优化？dfs,bfs...
        for i in range(row_bound):
            for j in range(col_bound):
                # 粗判断，减小计算量
                if i < small_radius_on_grid/1.4142 and j > (col_bound-small_radius_on_grid/1.4142):
                    continue
                elif min_radius_2 < ((j-row_bound)**2+i**2) < max_radius_2:
                    self.kernel[i, j] = 1
        # 翻转堆叠完圆
        self.kernel = np.hstack((self.kernel, np.fliplr(self.kernel)))
        self.kernel = np.vstack((np.flipud(self.kernel), self.kernel))

    def _points2grid(self, points: Array) -> Array:
        """将点划分进网格中，包含点的网格位置值为1

        - 参数:\n
            points - 待划分点\n
        - 返回:\n
            划分后的网格数组
        """
        bottomleft = [points[:, 0].min()-self.radius,
                      points[:, 1].min()-self.radius]
        upperright = [points[:, 0].max()+self.radius,
                      points[:, 1].max()+self.radius]
        # 获取边界
        self.max_row, self.max_col = int(
            (upperright[1]-bottomleft[1])//self.grid_len), int((upperright[0]-bottomleft[0])//self.grid_len)
        # 创建网格
        grid = np.zeros((self.max_row, self.max_col))
        for point in points:
            # 原点会出现数值计算问题，直接避开
            if point[0] != 0 and point[1] != 0:
                # 从=1改为+=1累计点的数目减小grid_len带来的点的精度损失
                grid[int(self.max_row-(point[1]-bottomleft[1])//self.grid_len) -
                    1, int((point[0]-bottomleft[0])//self.grid_len)-1] += 1
        
        return grid

    def __call__(self, points: Array) -> Array:
        """开始拟合圆

        - 参数:
            points - 待拟合点
            min_score - 用于筛选圆心的得分阈值\n
            estimate_dist - estimation of target's distance.
        - 返回:
            拟合圆的中心坐标列表
        """
        rcmsg = globalv.get_var('rcmsg')
        convert = np.pi/180
        close_dist = self.radius // self.grid_len
        # 卷积匹配圆
        ret = cv.filter2D(self._points2grid(points), -1, self.kernel)
        # 可疑点
        # 通过阈值过滤点，耗时60ms，可尝试torch计算
        i = np.arange(ret.shape[0])
        j = np.arange(ret.shape[1])       
        i = -((i+1-self.max_row)*self.grid_len-points[:, 1].min()+self.radius/2)
        j = (j+1)*self.grid_len+points[:, 0].min()-self.radius
        i, j = np.power(i, 2), np.power(j, 2)
        i, j = np.repeat(i[:, np.newaxis], ret.shape[1], 1), np.repeat(j[np.newaxis, :], ret.shape[0], 0)
        dist = np.sqrt(i + j)
        pn = 2 * np.arcsin(self.radius/dist) / convert / 0.25 - 2
        # clip at ten
        pn[pn < 10] = 10
        # -----------优化界限-----------
        # 匹配分数筛选，此处排序后可增大精确度，0.8为可调参数
        idxs = np.argwhere(ret >= pn*0.8)

        centers = []
        # 遍历所有可疑点
        for id in idxs:
            point = id
            close = False
            # 判断两个圆心是否过近
            for pre_point in centers:
                if point[0] - pre_point[0] < close_dist and point[1]-pre_point[1] < close_dist:
                    close = True
                    break
            if not close:
                centers.append([point[0], point[1]])
        if len(centers) == 0: 
            rcmsg.info('未找到圆')
            return centers
        # 从网格中恢复坐标真实值
        centers = np.asarray(centers, dtype=np.float)
        centers[:, 0] = -((centers[:, 0]+1-self.max_row)*self.grid_len-points[:, 1].min()+self.radius/2)
        centers[:, 1] = (centers[:, 1]+1)*self.grid_len+points[:, 0].min()-self.radius
        
        return centers