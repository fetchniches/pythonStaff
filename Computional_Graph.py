from typing import Callable, Any, Optional ,Sequence, Union

class GraphNode(object):

    def __init__(self, prior_node: Optional[Sequence] = None, next_node: Optional[Sequence] = None, 
                func: Callable = lambda x:x, *args, **kargs):
        """
        初始化结点

        参数
        -----
        prior_node - Sequence[GraphNode]
            指向该结点的结点
        func - Callable
            该结点的计算函数，默认函数为: f(x) = x
        next_node - Sequence[GraphNode]
            该结点的指向的结点
        """
        super().__init__()
        self.pn = [] if prior_node is None else prior_node
        self.nn = [] if next_node is None else next_node
        self.func = func
        self.val = None

    def __call__(self, *values):
        "计算结果并返回，存于val属性中"
        self.val = self.func(*values)
        return self.val

    def set_val(self, val):
        "修改val属性"
        self.val = val
        return val

    def get_val(self):
        "获取val属性"
        return self.val
 
def _seq_check(target: object):
    "检查是否为Sequence"
    if type(target) == list or type(target) == tuple:
        return True
    else:
        return False

def set_node_val(*nodes: GraphNode, values: Union[Any, Sequence[Any]]):
    """
    为结点赋初值

    参数
    -----
    nodes - GraphNode
        待初始化结点
    values - Union[int, Sequence]
        用于赋值，若为列表或元组则按对应顺序赋值
    """
    if _seq_check(values) and len(values) != len(nodes):
        raise ValueError('The values does not match to the nodes!')
    # 初值个数与结点数不匹配
    elif not _seq_check(values):
        values = [values]*len(nodes)
    # 扩展values
    for node, value in zip(nodes, values):
        node.set_val(value)

def add_node(*nodes: GraphNode):
    """
    向计算图添加结点，图以字典形式存储，字典键为结点对象的实例，值为指向结点的列表

    参数
    -----
    nodes - Graph
        待添加的结点集合，未检查重复添加结点

    返回
    -----
    len(nodes) - 总共添加结点的数量
    """
    for node in nodes:
        for _next in node.nn:
            _next.pn.append(node)
            # 将下一结点的prior结点自身
        # 添加后一结点
        for _prior in node.pn:
            _prior.nn.append(node)
            # 将上一结点的next加上自身
        # 遍历前一结点
    return len(nodes)

def compute(node: GraphNode):
    "计算某一结点的值"
    prior_val = []
    if not len(node.pn):
        if node.val is None:
            raise ValueError('The value hadn\'t been initialized!')
        # 叶结点未赋值
        return node.val
    # 返回叶结点的值
    for _prior in node.pn:
        prior_val.append(compute(_prior))
    ret = node(*prior_val)
    return ret


if __name__ == '__main__':
    class number(GraphNode):
    
        def __init__(self, number: int, *args, **kargs):
            super().__init__(*args, **kargs)
            self.num = number
            super().set_val(number)

        def __add__(self, x):
            # 创建结点
            new_node = number(self.num+x.num, func=lambda x, y:x+y, prior_node=[self, x])
            add_node(new_node)
            return new_node

        def __sub__(self, x):
            new_node = number(self.num-x.num, func=lambda x, y:x-y, prior_node=[self, x])
            add_node(new_node)
            return new_node

        def __str__(self):
            return str(self.num)

    n1 = number(2)
    n2 = number(3)
    n3 = n1-n2
    n4 = n1 -n2 + n1
    print(compute(n4))
    