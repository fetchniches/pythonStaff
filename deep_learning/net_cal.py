import math

def padding_work(insize: int, kernel: int, stride: int):
    """计算对应的输入尺寸下不变卷积的填充参数 (针对pytorch)

    - 参数: \n
        insize - 输入尺寸，正方形尺寸的边长\n
        kernel - 卷积核大小\n
        stride - 卷积步长 \n
    - 返回:\n
        填充参数, 返回的不一定为整数
    """
    padding = ((insize/stride-1)*stride+1+(kernel-1)-insize)/2 
    print(padding)
    return padding

def outsize_pt(insize: int, kernel: int, stride: int, padding: int):
    """计算给定参数下卷积后的输出尺寸

    - 参数: \n
        insize - 输入尺寸\n 
        kernel - 卷积核大小\n
        stride - 卷积步长 \n
        padding - 填充数量
    
    - 返回:\n
        返回经过卷积后输出的尺寸
    """
    ret = math.floor((insize+2*padding-(kernel-1)-1)/stride + 1)
    print(ret)
    return ret

