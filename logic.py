from Computional_Graph import *

def _or(x: int, y: int):
    return x+y if x+y<1 else 1

def _and(x: int, y: int):
    return 1 if x+y==2 else 0

def _xor(x: int, y: int):
    return 0 if x==y else 1

def _not(x: int):
    return 1-x

def seq_to_str(recv: Sequence):
    ret = ''
    for item in recv:
        ret += str(item)
    return ret

def str_to_seq(recv: str):
    ret = []
    for char in recv:
        ret.append(int(char))
    return ret

def binary_str(num: int, bits: int):
    ret = str(bin(num)[2:])
    if len(ret) < bits:
        add = ''.join(['0']*(bits-len(ret)))
        ret = add+ret
    return ret

class Logic(GraphNode):
    
    def __init__(self, number: int = 0, **kargs):
        super().__init__(**kargs)
        self.number = 1 if number!=0 else 0
        super().set_val(self.number)
        # 记录当前逻辑数的ID，用于表达式中
    
    def set_val(self, val):
        self.number = val
        return super().set_val(val)

    def x_convert(self, x: Any):
        "将x转换为Logic实例"
        return x if isinstance(x, Logic) else Logic(1) if x!=0 else Logic(0)

    def __or__(self, x):
        "或运算"
        x = self.x_convert(x)
        num = _or(x.number, self.number)
        ret = Logic(num, prior_node = [self, x], func=_or)
        add_node(ret)
        return ret
    
    def __and__(self, x):
        "与运算"
        x = self.x_convert(x)
        num = _and(x.number, self.number)
        ret = Logic(num, prior_node = [self, x], func=_and)
        add_node(ret)
        return ret
    
    def __xor__(self, x):
        "异或运算"
        x = self.x_convert(x)
        num = _xor(x.number, self.number)
        ret = Logic(num, prior_node = [self, x], func=_xor)
        add_node(ret)
        return ret

    def n(self):
        "非运算"
        num = _not(self.number)
        ret = Logic(num, prior_node = [self], func=_not)
        add_node(ret)
        return ret
    
    def __str__(self):
        return str(self.number)

def MultiLN(total: int, *numbers: Optional[int]):
    """
    创建多个逻辑数对象

    参数
    -----
    total - int
        需要创建的对象个数
    numbers - Sequence[int]
        对应逻辑数的初值，当未指定时自动赋值为0

    返回
    -----
    lns - list
        逻辑数对象构成的列表
    """
    lns = []
    if not len(numbers):
        numbers = [0]*total
    for num in numbers:
        lns.append(Logic(num))
    return lns

class StateTransformer(object):

    def __init__(self, exps: Sequence[Logic], vars: Sequence[Logic]):
        """
        初始化状态图转换器

        参数
        -----
        exps - Sequence
            表达式构成的列表或元组
        vars - Sequence
            表达式包含的变量列表或元组
        """
        super().__init__()
        self.exps = exps
        self.vars = vars
        self.all_states = self._total_state()

    def _total_state(self):
        "获取所有状态"
        total = []
        for i in range(2**len(self.vars)):
            total.append(str_to_seq(binary_str(i, len(self.vars))))
        return total

    def forward(self, reverse: bool = False):
        """
        状态图计算

        参数
        -----
        reverse - bool
            设置变量显示顺序，默认从0~N
        """
        access_states = []
        non_access_states = self.all_states.copy()
        # 初始化未访问状态与已访问状态列表
        while len(non_access_states):
            self._loop(access_states, non_access_states,init_vals=non_access_states[0], reverse=reverse)

    def _loop(self, access_states: list, non_access_states: list, init_vals: Sequence[int], reverse: bool): 
        """
        循环计算状态图

        参数
        -----
        access_states - list
            已经访问过的状态列表
        non_access_states - list
            未访问的状态列表
        init_vals - Sequence
            初始状态
        reverse - bool
            设置变量显示顺序，默认从0~N
        """
        arrow = '  ->  '
        print('-'*25)
        step = -1 if reverse else 1
        print(seq_to_str(init_vals)[::step]+arrow, end='')
        current_state = init_vals
        access_states.append(init_vals)
        non_access_states.remove(init_vals)
        while True:
            set_node_val(*self.vars, values=current_state)
            # 设置值
            current_state = []
            # 保存当前状态
            for exp in self.exps:
                current_state.append(compute(exp))
            print(seq_to_str(current_state)[::step])
            try:
                non_access_states.remove(current_state)
            except ValueError:
                pass
            if access_states.count(current_state):
                break
            else:
                access_states.append(current_state)
            # 判断跳出循环条件
            print(seq_to_str(current_state)[::step]+arrow, end='')


if __name__ == '__main__':
    q0, q1, q2, q3 = MultiLN(4)
    exp0 = q0.n()
    exp1 = (q0.n()&q1.n()&(q2|q3))|q0&q1
    exp2 = q0.n()&q2.n()&q3|(q0|q1)&q2
    exp3 = q0.n()&q1.n()&q2.n()&q3.n()|(q0&q3)
    st = StateTransformer([exp0, exp1, exp2, exp3], [q0, q1, q2, q3])
    st.forward(reverse=True)




