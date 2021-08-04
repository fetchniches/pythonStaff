import numpy as np 
from typing import Tuple
import matplotlib.pyplot as plt
import random

def plot3d(data: np.ndarray, **kwargs):
    """绘制3维坐标点

    - 参数:\n 
        data - nx3维数据, n为待绘制的点个数\n 
        kwargs - 这些参数将直接传递给绘图函数
    """
    ax = plt.subplot(projection='3d')
    ax.scatter3D(data[:, 0], data[:, 1], data[:,2], **kwargs)
    plt.show()

class RandomPoint(object):
    """该类用于生成指定结构的随机的三维坐标点
    """
    def __init__(self, total_num: int, point_num: int,bound: Tuple[int], symet: int) -> None:
        """
        - 参数:\n 
            total_num - 生成结构数目\n 
            point_num - 对应结构需要生成的点个数\n 
            bound - 对应生成点的边界\n 
            symet - 对称性，有2，4，8对称格式\n 
        """
        super().__init__()
        self.snum = total_num
        # 随机点生成结构数
        self.pnum = point_num
        # 点个数
        self.bound = bound
        # 边界
        self.sym = symet
        # 对称
        self.missing = abs(int(self.pnum//self.sym)*self.sym - self.pnum)
    
    def generate_points(self):
        "生成self.pnum/self.sym个点"
        if self.sym == 2:
            ls_num = [1, 1, 2]
        elif self.sym == 4:
            ls_num = [1, 2, 2]
        elif self.sym == 8:
            ls_num = [2, 2, 2]
        for i in range(int(self.pnum//self.sym)):
            x = random.uniform(0, 1)*(self.bound[1]-self.bound[0])/ls_num[0]
            y = random.uniform(0, 1)*(self.bound[1]-self.bound[0])/ls_num[1]
            z = random.uniform(0, 1)*(self.bound[1]-self.bound[0])/ls_num[2]
            self.points.append((x+self.bound[0], y+self.bound[0], z+self.bound[0]))

    def duplicate_points(self, sym: int):
        current = self.points.copy()
        if sym == 2:
            for point in current:
                dis = -point[2]
                self.points.append((point[0], point[1], point[2]+dis*2))
        elif sym == 4:
            for point in current:
                dis = -point[1]
                self.points.append((point[0], point[1]+dis*2, point[2]))
            self.duplicate_points(int(sym/2))
        elif sym == 8:
            for point in current:
                dis = -point[0]
                self.points.append((point[0]+dis*2, point[1], point[2]))
            self.duplicate_points(int(sym/2))

    def get_points(self):
        self.struct = []
        for s in range(self.snum):
            self.points = []
            self.generate_points()
            self.duplicate_points(self.sym)
            if self.missing:
                for i in range(self.missing):
                    x = random.uniform(0, 1)*(self.bound[1]-self.bound[0])
                    y = random.uniform(0, 1)*(self.bound[1]-self.bound[0])
                    z = random.uniform(0, 1)*(self.bound[1]-self.bound[0])
                    self.points.append((x+self.bound[0], y+self.bound[0], z+self.bound[0]))
            self.struct.append(self.points)
        return np.asarray(self.struct, dtype=np.float32).reshape(self.snum, -1)


a = RandomPoint()