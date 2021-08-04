# 执行各式图片的相关操作
# 计划添加批量处理图片功能 
from typing import Dict, List, NewType, Optional, Sequence, Tuple, Union
import cv2 as cv
import numpy as np
from PIL import Image
import random


Imge = NewType('fetImage', np.ndarray)


class ImageProcess(object):

    def __init__(self) -> None:
        super().__init__()
        color = ['black', 'gray', 'white', 'red', 'orange',
                 'yellow', 'green', 'cyan', 'blue', 'purple']
        hsv_val = [[(0, 0, 0), (180, 255, 46)], [(0, 0, 46), (180, 43, 220)],
                   [(0, 0, 221), (180, 30, 255)], [
            (156, 43, 46), (180, 255, 255)],
            [(11, 43, 46), (25, 255, 255)], [
            (26, 43, 46), (34, 255, 255)],
            [(35, 43, 46), (77, 255, 255)], [(78, 43, 46), (99, 255, 255)],
            [(100, 43, 46), (124, 255, 255)], [(125, 43, 46), (155, 255, 255)]]
        self.clr_dict = {}
        for clr, val in zip(color, hsv_val):
            self.clr_dict.setdefault(clr, val)
        # 创建颜色索引

    def load_image(self, image: Union[str, np.ndarray], *args, **kargs) -> None:
        """加载图片
        - 参数:\n 
            img_path - 待载入图片的路径
        """
        if type(image) == str:
            self.image = cv.imread(image, *args, **kargs)
        elif type(image) == np.ndarray:
            self.image = image
        else:
            raise ValueError('Wrong type for image!')

    def create_color_mask(self, mask_color: Sequence,
        extra_operation: Optional[Sequence] = None
    ) -> Imge:
        """创建对应HSV色彩的掩膜

        - 参数:\n 
            mask_color - 掩膜保留的HSV色彩，元组或列表类型，如:((11, 43, 46), (25, 255, 255)).\n
            extra_operation - 对创建后的掩膜进行额外的形态学操作，参数为列表，包含[操作类型, 内核大小, 迭代次数]，操作类型参数有: 
            'open', 'close', 'dilate', 'erode'，内核大小是内含两个整数的元组，迭代次数是操作反复执行的次数\n

        - 返回:\n
            创建后的掩膜，ndarray类型的数据
        """
        clr_lower = mask_color[0]
        clr_upper = mask_color[1]
        img = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(img, clr_lower, clr_upper)
        if extra_operation is not None:
            if extra_operation[0].upper() == 'OPEN':
                mask = cv.morphologyEx(
                    mask, cv.MORPH_OPEN, extra_operation[1], iterations=extra_operation[2])
            elif extra_operation[0].upper() == 'CLOSE':
                mask = cv.morphologyEx(
                    mask, cv.MORPH_CLOSE, extra_operation[1], iterations=extra_operation[2])
            elif extra_operation[0].upper() == 'DILATE':
                mask = cv.dilate(mask, kernel=extra_operation[1], iterations=extra_operation[2])
            elif extra_operation[0].upper() == 'ERODE':
                mask = cv.erode(mask, kernel=extra_operation[1], iterations=extra_operation[2])
        
        return mask


    def pil2cv(self):
        "将PIL格式图片转换为ndarray格式"
        img = cv.cvtColor(np.asarray(self.image), cv.COLOR_RGB2BGR)
        return img

    def cv2pil(self):
        "将ndarray格式图片转换为PIL格式"
        img = Image.fromarray(cv.cvtColor(self.image, cv.COLOR_BGR2RGB))
        return img


def cntsort(cnts: Sequence) -> List:
    """轮廓面积大小倒序排序

    - 参数:\n
        cnts - 由opencv找到的轮廓列表 \n 

    - 返回:\n
        大小倒序排序的轮廓列表
    """
    sorted_cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    # 根据轮廓面积从小到大排序
    return sorted_cnts


def video2img(source: str, img_per_frame: int, img_file: str) -> None:
    """从视频中读取图片

    - 参数:\n 
        source - 视频文件的路径\n
        img_per_frame - 截取图片间隔的帧数\n 
        img_file - 图片输出的路径    
    """
    cap = cv.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError('Failed to open the video!')
    i = 1
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('Finished!')
            break
        if not i % img_per_frame:
            cv.imwrite(img_file + '{}.jpg'.format(int(i/img_per_frame)), frame)
        print('Reading the', i, 'th frame.')
        i += 1


def read_HSV_color(img: Imge):
    """以灰度图显示HSV分量

    - 参数:\n 
        img - 读取图片
    """
    out_img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # 将图片转换为灰度图
    hsvChannels = cv.split(out_img_HSV)
    # 将HSV格式的图片分解为3个通道
    cv.namedWindow("Hue", 2)
    # 创建一个窗口
    cv.imshow('Hue', hsvChannels[0])
    # 显示Hue分量
    cv.namedWindow("Saturation", 2)
    # 创建一个窗口
    cv.imshow('Saturation', hsvChannels[1])
    # 显示Saturation分量
    cv.namedWindow("Value", 2)
    # 创建一个窗口
    cv.imshow('Value', hsvChannels[2])
    # 显示Value分量
    cv.waitKey(0)
    cv.destroyAllWindows()


def show_image(*images) -> None:
    "显示当前加载的图片"
    i = 1
    for img in images:
        cv.imshow('Image-{}'.format(i), img)
        i += 1
    cv.waitKey(0)
    cv.destroyAllWindows()


def MotionBlur(img: Imge, size: int) -> Imge:
    """给图片添加动态模糊

    - 参数:\n 
        img - 目标图片\n
        size - 卷积的内核大小，越大动态模糊越剧烈\n

    - 返回:\n 
        处理后的图片
    """
    kernel = np.zeros((size, size))
    for i in range(size):
        kernel[i, i] = 1/size
    img = cv.filter2D(img, -1, kernel)
    return img


def get_rect_contours(cnts: List, rect_margin: Optional[int] = None) -> List[int]:
    """获取轮廓矩形近似的x,y,w,h

    - 参数:\n 
        src - 目标图片\n
        cnts - 目标图片的轮廓列表\n
        rect_margin - 矩形获取的间隔，太过靠近的矩形会被丢弃
    
    - 返回:\n
        形如[(x, y, w, h), ...]的列表，与原轮廓列表中的轮廓一一对应
    """
    rect = []
    for cnt in cnts:
        close = 0
        x, y, w, h = cv.boundingRect(cnt)
        if rect_margin:
            for xp, yp, _, _ in rect:
                if (x-xp)**2+(y-yp)**2 < rect_margin**2:
                    close = 1
                    break
        if close == 0:
            rect.append((x, y, w, h))
    return rect


def draw_rect(background: Imge, rect: List, color: Tuple = (0, 0, 255), thickness: int = 10):
    if rect is not None:
        for x, y, w, h in rect:
            cv.rectangle(background, (x, y), (x+w, y+h), color, thickness)


def draw_rect_contours(background: Imge, cnts: List,
    color: Tuple = (0, 0, 255), thickness: int = 10, draw_margin: Optional[int] = None
) -> None:
    """绘制轮廓的直角矩形

    - 参数:\n 
        background - 指定矩形绘制的图片\n 
        cnts - 待绘制的轮廓\n 
        color - RGB轮廓色彩\n
        thickness - 矩形厚度\n
        draw_margin - 矩形绘制的间隔，太过靠近的矩形会被排除
    """
    pos = get_rect_contours(background, cnts, draw_margin)
    for x, y, w, h in pos:
        cv.rectangle(background, (x, y), (x+w, y+h), color, thickness)


def random_color(mode: str = 'BGR') -> Tuple:
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    if mode.upper() == 'BGR':
        return (b, g, r)
    elif mode.upper() == 'RGB':
        return (r, g, b)


def iamge_combinition(fst_img: Imge, other_img: Imge, mode: str) -> Imge:
    """合并图片

    - 参数:\n 
        mode - 图片合成模式，'r'即横向插入模式，'c'即纵向插入模式
    - 返回:\n
        合并后的图片
    """
    row1, col1, chan1 = fst_img.shape
    row2, col2, chan2 = other_img.shape
    if mode == 'r':
        ret = np.zeros([row1 + row2, max([col1, col2]),
                        max([chan1, chan2])], dtype=np.uint8)
        ret[:row1, :col1] += fst_img
        ret[row1:row1+row2, :col2] += other_img
    elif mode == 'c':
        ret = np.zeros([max(row1, row2), col1 + col2,
                        max([chan1, chan2])], dtype=np.uint8)
        ret[:row1, :col1] += fst_img
        ret[:row2, col1:col1+col2] += other_img

    return ret


def image_resize(image, rtimes: float = 1, ctimes: float = 1, method: int = 0) -> Imge:
    """图像缩放

    - 参数:\n
        rtimes -  纵向缩放倍数\n
        ctimes - 表示横向缩放倍数\n
        method - 缩放方式0-4，推荐使用1即线性缩放\n

    - 返回:\n
        缩放操作后的图像
    """
    width, height = image.shape[:2]
    inter_method = [cv.INTER_NEAREST, cv.INTER_LINEAR,
                    cv.INTER_AREA, cv.INTER_CUBIC, cv.INTER_LANCZOS4]
    ret = cv.resize(image, (int(height*ctimes), int(width*rtimes)),
                    interpolation=inter_method[method])
    return ret


def get_min_rect(cnts: List) -> List:
    """获取最小矩形轮廓

    - 参数:\n 
        cnts - 目标图片的轮廓列表\n 
    
    - 返回:\n
        形如[(w, h), ...]的列表，与原轮廓列表中的轮廓一一对应
    """
    rects = [] 
    for cnt in cnts:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        distance = []
        for idx1 in range(len(box)):
            for idx2 in range(len(box[idx1+1:])):
                distance.append(np.sqrt((box[idx1][0]-box[idx2+idx1+1][0])**2+(box[idx1][1]-box[idx2+idx1+1][1])**2))
        distance.sort()
        w = (distance[0]+distance[1])/2
        h = (distance[2]+distance[3])/2
        rects.append((w, h))
    return rects


def draw_min_rect(background: Imge, cnts: List, color: Tuple = (0, 0, 255), 
    thickness: int = 10):
    """绘制轮廓的最小矩形

    - 参数:\n 
        background - 指定矩形绘制的图片\n 
        cnts - 待绘制的轮廓\n 
        color - RGB轮廓色彩\n
        thickness - 矩形厚度\n
    """
    boxes = []
    for cnt in cnts:
        rect = cv.minAreaRect(cnt)
        boxes.append(np.int0(cv.boxPoints(rect)))
    for box in boxes:
        cv.drawContours(background, [box], 0, color, thickness)


def get_simple_contours(image: Imge, filter_area: Optional[Sequence[int]] = None) -> List:
    """最简单的方式获取轮廓

    - 参数:\n
        image - 处理的图片
        filter_area - 用于过滤小面积的轮廓，例如: 使用(lower, upper)过滤面积在lower至upper间的轮廓或[-1, upper]过滤小于upper的轮廓
    """
    org_cnts, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # 获取原始轮廓
    cnts = []
    if filter_area is not None:
        lower, upper = filter_area
        for cnt in org_cnts:
            area = cv.contourArea(cnt)
            if lower == -1 and upper == -1:
                raise ValueError("Can't make two barriers to be -1!")
            elif upper == -1:
                if area > lower:
                    cnts.append(cnt)
            elif lower == -1:
                if area < upper:
                    cnts.append(cnt)
            # 面积下限为-1时
            # 面积上限为-1时
            else:
                if upper > area > lower:
                    cnts.append(cnt)
    # 筛选轮廓
    else:
        cnts = org_cnts
    return cnts


