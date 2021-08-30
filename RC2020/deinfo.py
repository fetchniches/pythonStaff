import datetime
import time
from typing import Any
import inspect
import re

def _gettime():
    t = list(time.localtime())[:6]
    return str(datetime.datetime(*t))

def show_msg(msg_type: str, msg: str):
    print(_gettime()+' ['+msg_type.upper()+'] '+msg)


class debugInfo(object):

    def __init__(self, debug: bool):
        super().__init__()
        self.debug = debug

    def info(self, msg: str):
        "输出普通信息"
        if self.debug:
            show_msg('info', msg)

    def warning(self, msg: str):
        "输出警告"
        if self.debug:
            show_msg('warning', msg)

    def variable(self, *vars: Any):
        "输出变量信息"
        if self.debug:
            type_re = re.compile(r"'.+'")
            callers_local_vars = inspect.currentframe().f_back.f_locals.items()
            # 获取所有变量名
            output_vars = []
            for var in vars:
                for var_name, var_val in callers_local_vars:
                    if var_val is var:
                        break
                output_vars.append("{} = {}".format(var_name, var))                
                #output_vars.append("{}: {} = {}".format(var_name, type_re.search(str(type(var))).group().strip("'"), var))
            show_msg('variable', ', '.join(output_vars))