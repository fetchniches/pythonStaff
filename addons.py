# 代码插入
from typing import Sequence
from types import FunctionType

class Addons(object):
    "代码插入类"
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
        """调用所有函数

        - 参数:\n 
            params - 所有函数的参数\n 
        
        - 返回:\n 
            所有函数的返回值构成的字典，函数名(字符串)为键
        """
        ret = {}
        # 存储函数返回值
        for func in self.functions.keys():
            current_params = {}
            args = self.functions[func]
            # 获取当前函数参数列表
            for arg in args:
                current_params.setdefault(arg, params[arg])
            # 获取参数
            ret.setdefault(func.__name__, func(**current_params))
            # 执行函数并将返回值存入字典
        return ret


# add codes below
