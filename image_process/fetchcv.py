from typing import Callable, Dict, List, Literal, NewType, Optional, Sequence, Tuple, Union, overload
import cv2 as cv
import numpy as np
from PIL import Image
import random
from enum import Enum
import os
import math

class Color(Enum):
    """
    HSV Value
    """
    BLACK = [(0, 0, 0), (180, 255, 46)]
    GRAY = [(0, 0, 46), (180, 43, 220)]
    WHITE = [(0, 0, 221), (180, 30, 255)]
    RED = [(156, 43, 46), (180, 255, 255)]
    ORANGE = [(11, 43, 46), (25, 255, 255)]
    YELLOW = [(26, 43, 46), (34, 255, 255)]
    GREEN = [(35, 43, 46), (77, 255, 255)]
    CYAN = [(78, 43, 46), (99, 255, 255)]
    BLUE = [(100, 43, 46), (124, 255, 255)]
    PURPLE = [(125, 43, 46), (155, 255, 255)]


class ContoursSearch(object):

    def __init__(self):
        super().__init__()
        # in process
        self.in_filters = []
        # after process
        self.af_filters = []

    def addFilters(self, *filters: Callable):
        """
        添加轮廓过滤器

        参数
        -----
        filters: Callable
            过滤器，函数名需要以`IN`或`AF`结尾，以`IN`结尾的作用于单个轮廓，在遍历时调用，只接收单个轮廓为参数，
            并返回`True`或`False`以判断是否保留。以`AF`结尾的，参数接收整个轮廓集合，并返回滤过的轮廓。

        过滤器函数示例
        -----
        >>> def filter_IN(cnt: List) -> bool:
        ...     area = cv.contourArea(cnt)
        ...     return True if area < 200 else False
        >>> ContourLocator.addFilters(filter_IN)
        """
        for filter in (filters):
            if filter.__name__[-2:] == 'IN':
                self.in_filters.append(filter)
            elif filter.__name__[-2:] == 'AF':
                self.af_filters.append(filter)
            else:
                raise NameError('function name gets wrong, must end with \'IN\' or \'AF\'')

    def __call__(self, image):
        original_cnts, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        filted_cnts = []
        # 遍历单个轮廓的部分
        for cnt in original_cnts:
            # 通过的过滤器数目
            pass_numbers = sum([filter(cnt) for filter in self.in_filters])
            if pass_numbers == len(self.in_filters):
                filted_cnts.append(cnt)
        # 遍历整体轮廓
        for filter in self.af_filters:
            filted_cnts = filter(filted_cnts)
        return filted_cnts


def areaFilter(min_area: int, max_area: int):
    """
    通过面积过滤轮廓

    参数
    ------
    min_area: int
        最小面积阈值
    max_area: int
        最大面积阈值，可设置为np.inf或math.inf
    """
    def area_check_IN(cnt):
        area = cv.contourArea(cnt)
        return True if min_area < area < max_area else False
    return area_check_IN

def intervalFilter(interval: int):
    """
    通过轮廓间隔过滤，过于靠近的轮廓将被忽略

    参数
    -----
    interval: int
        间隔值，单位：像素。
    """
    def interval_check_AF(cnts: List):
        ret = []
        centers = []
        for cnt in cnts:
            close = False
            x, y, w, h = cv.boundingRect(cnt)
            for bbox in centers:
                if (bbox[0]-x-w/2)**2+(bbox[1]-y-h/2)**2 < interval**2:
                    close = True
                    break
            if not close:
                ret.append(cnt)
                centers.append((x+w/2, y+h/2))
        return ret
    return interval_check_AF


def getRects(cnts: List):
    """
    获取轮廓对应的边界框，与图像平行，不会是面积最小的边界框
    
    参数
    -----
    cnts: List
        获取得到的轮廓
    
    示例
    -----
    >>> cnts, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    >>> getRects(cnts)
    """
    return [cv.boundingRect(cnt) for cnt in cnts]

def drawContoursRects(image: Union[os.PathLike, np.ndarray], cnts: List, color: Tuple[int] = (0, 0, 255), thickness: int = 5):
    """
    绘制边界框，与图像边界平行

    参数
    -----
    image: Union[os.PathLike, np.ndarray]
        图像数组或图像路径
    cnts: List
        轮廓列表
    color: Tuple
        边界框色彩
    thickness: int
        边界框线条粗细，单位：像素
    """
    image = _loadImage(image)
    for x, y, w, h in getRects(cnts):
        cv.rectangle(image, (x, y), (x+w, y+h), color, thickness)

def drawMinimumContours(image: Union[os.PathLike, np.ndarray], cnts: List, color: Tuple[int] = (0, 0, 255), thickness: int = 5):
    """
    绘制最小的边界框

    参数
    -----
    image: Union[os.PathLike, np.ndarray]
        图像数组或图像路径
    cnts: List
        轮廓列表
    color: Tuple
        边界框色彩
    thickness: int
        边界框线条粗细，单位：像素
    """
    bboxes = [np.int32(cv.boxPoints(cv.minAreaRect(cnt))) for cnt in cnts]
    cv.drawContours(image, bboxes, -1, color, thickness)

def _loadImage(img: Union[os.PathLike, np.ndarray], *args, **kwargs):
    if isinstance(img, str):
        return cv.imread(img)
    elif isinstance(img, np.ndarray):
        return img
    else:
        raise ValueError('image type(s) must be array-like or path.')

def imageCombination(fst_img: Union[os.PathLike, np.ndarray], other_img: Union[os.PathLike, np.ndarray], mode: str):
    """合并图片

    - 参数:\n 
        mode - 图片合成模式，'r'即横向插入模式，'c'即纵向插入模式
    - 返回:\n
        合并后的图片
    """
    fst_img, other_img = _loadImage(fst_img), _loadImage(other_img)
    row1, col1, chan1 = fst_img.shape
    row2, col2, chan2 = other_img.shape
    if mode.lower() == 'r':
        ret = np.zeros([row1 + row2, max([col1, col2]), max([chan1, chan2])], dtype=np.uint8)
        ret[:row1, :col1] += fst_img
        ret[row1:row1+row2, :col2] += other_img
    elif mode.lower() == 'c':
        ret = np.zeros([max(row1, row2), col1 + col2, max([chan1, chan2])], dtype=np.uint8)
        ret[:row1, :col1] += fst_img
        ret[:row2, col1:col1+col2] += other_img
    return ret

def imageResize(image: Union[os.PathLike, np.ndarray], rtimes: float = 1, ctimes: float = 1, method: int = 0):
    """图像缩放

    - 参数:\n
        rtimes -  纵向缩放倍数\n
        ctimes - 表示横向缩放倍数\n
        method - 缩放方式0-4，推荐使用1即线性缩放\n

    - 返回:\n
        缩放操作后的图像
    """
    image = _loadImage(image)
    width, height = image.shape[:2]
    inter_method = [cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_AREA, cv.INTER_CUBIC, cv.INTER_LANCZOS4]
    return cv.resize(image, (int(height*ctimes), int(width*rtimes)), interpolation=inter_method[method])

def createColorMask(image: Union[os.PathLike, np.ndarray], 
    color: Union[Color, Sequence], extra: Optional[Sequence] = None):
    """
    创建掩膜
    """
    mask: np.ndarray = cv.inRange(cv.cvtColor(_loadImage(image), cv.COLOR_BGR2HSV), *color.value)
    if extra is not None:
        if extra[0].upper() == 'OPEN':
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, extra[1], iterations=extra[2])
        elif extra[0].upper() == 'CLOSE':
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, extra[1], iterations=extra[2])
        elif extra[0].upper() == 'DILATE':
            mask = cv.dilate(mask, kernel=extra[1], iterations=extra[2])
        elif extra[0].upper() == 'ERODE':
            mask = cv.erode(mask, kernel=extra[1], iterations=extra[2])
    return mask

def readHSVColor(image: Union[os.PathLike, np.ndarray]):
    """以灰度图显示HSV分量

    - 参数:\n 
        img - 读取图片
    """
    image = _loadImage(image)
    out_img_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # 将HSV格式的图片分解为3个通道
    hsvChannels = cv.split(out_img_HSV)
    
    cv.namedWindow("Hue")
    cv.imshow('Hue', hsvChannels[0])
    cv.namedWindow("Saturation")
    cv.imshow('Saturation', hsvChannels[1])
    cv.namedWindow("Value")
    cv.imshow('Value', hsvChannels[2])

    cv.waitKey(0)
    cv.destroyAllWindows()

def showImage(image: np.ndarray, title: Optional[str] = 'image'):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def video2image(source: os.PathLike, frame: int, output: os.PathLike):
    """
    从视频中读取图片，输出到指定文件夹

    参数
    -----
    source: PathLike
        视频源路径
    frame: int
        间隔帧数
    output: PathLike
        输出文件夹
    """
    cap = cv.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError('Failed to open the video.')
    i = 1
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('Finished!')
            break
        if not i%frame:
            cv.imwrite(os.path.join(output, '{}.jpg'.format(int(i/frame))), frame)
            print('Saved {}th picture'.format(int(i/frame)))
        i += 1

if __name__ == '__main__':
    ...