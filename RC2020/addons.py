# -*- coding: utf-8 -*-
from typing import Sequence
from types import FunctionType

from numpy.lib import math
import colorCV as ccv
import numpy as np 
import ldrprocess_new as ldr
import time
import matplotlib.pyplot as plt 
import USB as usb
import utils.globalvars as globalv

class Addons(object):

    def __init__(self, *functions: Sequence[FunctionType]):
        super().__init__()
        self.functions = {}
        # 初始化模块
        for func in functions:
            params_num = func.__code__.co_argcount
            # 获取函数参数个数
            params = func.__code__.co_varnames[:params_num]
            # 读取函数需要的参数列表
            self.functions.setdefault(func, params)
    
    def run(self, **params):
        ret = {}
        # 存储函数返回值
        for func in self.functions.keys():
            current_params = {}
            args = self.functions[func]
            # 获取当前函数参数列表
            for arg in args:
                try:
                    current_params.setdefault(arg, params[arg])
                except KeyError:
                    print('Warning - One argument is missing.')
            # 获取参数
            ret.setdefault(func.__name__, func(**current_params))
            # 执行函数并将返回值存入字典
        return ret


# codes added below:
bucket_width = 315  
bucket_height = 464
distance = 2000

def work_distance(x: int, w: int, h: int, ldport, c_tplt, dp):
    """
    根据雷达与视觉数据，计算并返回靠近视野中央的目标距离及角度

    参数
    -----
    x - int
        锚框的x坐标，单位像素
    w - int
        锚框的宽度，单位像素
    h - int
        锚框的高度
    ldport - Serial
        雷达的串口对象
    c_tplt - CircleTemplate
        匹配圆的模板
    dp - DynamicPlot
        用于绘制动态的雷达数据图

    返回
    -----
    dist - float | None
        目标距离，若无目标返回None
    deg - float | None
        目标角度，若无目标返回None
    """
    rcmsg = globalv.get_var('rcmsg')
    #获取每个检测目标的w与h
    f=510
    pred_dist=f/(h/bucket_height)*1.16
    degree = ((x + w / 2) / 640)*2-1
    depi=np.arctan(degree*0.7)
    # -60~60度
    pred_deg=depi/np.pi*180+90
    #pred_deg=((x + w / 2) / 640)*100+30
    # leida
    rcmsg.info('读取雷达数据...')
    data = usb.jieShou(ldport) 
    dist = deg = None
    try:
        dist, deg = lidar(data=data ,pred_dist=pred_dist, pred_deg=pred_deg, c_tplt=c_tplt, dp=dp)

    except TypeError as e:
        rcmsg.warning('引发异常'+str(e)+"，返回dist, deg")
        rcmsg.variable(dist, deg)
        return dist, deg

    return dist, deg

def send_info(vas, outg, outd, bits: int = 16):
    """
    发送数据

    参数
    -----
    vas - vSerial
        串口类
    outg - float
        目标角度，十进制
    outd - float
        目标距离，十进制
    bits - int
        传输位数，8或16位
    """
    rcmsg = globalv.get_var('rcmsg')
    rcmsg.info('准备发送串口数据')
    rcmsg.variable(outd, outg)
    try:
        maxval = 255 if bits == 8 else 65535
         

 #           si=0
#        outd=sum()
        send_dist = round((outd / 10000) * maxval)
        send_deg = round(((90-outg) / 180) * maxval)
    except TypeError:
        send_dist, send_deg = 0, round(maxval/2)
        # outd为NoneType时引发
    if send_dist == 0 and send_deg == round(maxval/2):
        vas.sendOneByteInHex('DF')
        vas.sendOneByteInHex('EF')
        rcmsg.info('未找到目标发送DF EF')
    else:
        vas.sendOneByteInHex('FF')
        vas.sendOneByteInIntEx(send_dist, bits)
        vas.sendOneByteInIntEx(send_deg, bits)
        vas.sendOneByteInHex('EF')
        rcmsg.info('找到目标，已发送数据如下 : ')
        rcmsg.variable(send_dist, send_deg)


def lidar(c_tplt, data, pred_dist: float, pred_deg: float, dp):
    """
    输入雷达数据进行匹配

    参数
    -----
    c_tplt - CircleTeplate
        圆匹配模板
    data - ndarray
        雷达的原始数据，极坐标形式
    pred_dist - float
        估计目标距离
    pred_deg - float
        估计目标角度
    
    返回
    -----
    dist - float
        计算得到的目标距离，若无匹配返回None
    deg - float
        计算得到的目标角度，若无匹配返回None
    """
    rcmsg = globalv.get_var('rcmsg')
    # 中心化
    pred_dist -= 152.5
    rcmsg.variable(pred_dist, pred_deg)
    # 过滤不需要的点
    poi = ldr.pointsSelector(data, pred_deg, 0.05)
    # 初始化返回值dist与deg
    dist = deg = None
    # 转换为直角坐标
    try:
        ldr.polarToCart(poi)
        ldr.polarToCart(data)
    # 数据点集poi或data为空时，返回None, None
    except IndexError:
        rcmsg.warning('雷达数据与视觉估测数据不匹配，返回dist, deg')
        rcmsg.variable(dist, deg)
        return dist, deg
    # 匹配圆，未匹配到圆时返回空列表。
    t1 = time.time()
    centers = c_tplt(poi)
    # centers = c_tplt(poi, estimate_dist=pred_dist)
    t2 = time.time()
    rcmsg.info('time spent {:.2f}ms'.format((t2-t1)*1e3))
    ldr.showMatRes(points=data, poi=poi, centers=centers, graph=True, dp=dp)
    # graph设置是否显示图像
    if len(centers):
        centers = sorted(centers, key=lambda x: abs(x[0]))
        dist = np.sqrt(centers[0][0]**2+centers[0][1]**2)
        deg = np.arctan(centers[0][1]/centers[0][0])*(180/np.pi)
    # 当未匹配到圆时，即centers为空列表时返回None
    else:
        rcmsg.info('算法未匹配到圆')
    return dist, deg