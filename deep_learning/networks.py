from typing import Sequence, Union
import torch
import torch.nn as nn
import torch.optim as optim
import re
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta

class Fetchnn(nn.Module):
    "快速搭建神经网络"
    def __init__(self, device: int, in_features: int, architect: Sequence, loss_func: nn.Module = nn.CrossEntropyLoss()):
        """

        - 参数:\n 
            device - 指定计算使用的设备\n
            in_features - 对应全连接网络的输入特征数或卷积网络的输入通道数\n
            architect - 神经网络的结构\n 
            loss_func - 神经网络使用的损失函数\n
            
        """
        super().__init__()
        self.device = device
        self.last_f = in_features
        self.loss_func = loss_func
        self.test_num = 0
        self.struct = nn.Sequential()
        print('Initializing Networks...')
        self._init_layers(architect)
        print(self.parameters)

    def forward(self, x):
        return self.struct(x)

    def _init_layers(self, architect: Sequence):
        "初始化网络"
        module_dict = {
            'conv1d':nn.Conv1d, 'conv2d':nn.Conv2d, 'conv3d':nn.Conv3d, 'linear':nn.Linear, 
            'dropout':nn.Dropout, 'dropout2d':nn.Dropout2d, 'dropout3d':nn.Dropout3d,
            'batchnorm1d':nn.BatchNorm1d, 'batchnorm2d':nn.BatchNorm2d, 'batchnorm3d':nn.BatchNorm3d,
            'maxpool1d':nn.MaxPool1d(2, 2), 'maxpool2d':nn.MaxPool2d(2, 2), 'maxpool3d':nn.MaxPool3d(2, 2), 'flatten':nn.Flatten(),
            'relu':nn.ReLU(inplace=True), 'softmax':nn.Softmax(dim=1), 'leakyrelu':nn.LeakyReLU(inplace=True), 'sigmoid':nn.Sigmoid()
        }
        conv_re = re.compile(r'conv', re.I)
        bn_re = re.compile(r'batchnorm', re.I)
        linear_re = re.compile(r'linear', re.I)
        mark = 0
        for layer in architect:
            mark += 1
        # 遍历结构的每一层
            if conv_re.match(layer[0]) is not None:
                name = layer[0]
                layer[0] = self.last_f
                if len(layer) == 3:
                    layer.append(1)    
                # 缺省参数stride
                if len(layer) == 4:
                    layer.append(int((layer[2]-1)/2))
                # 添加参数padding
                self.last_f = layer[1]
                # 更新当前的特征数
                current_layer = (name+'_'+str(mark), module_dict[name](*layer)) 
            # 匹配卷积层
            elif bn_re.match(layer[0]) is not None:
                name = layer[0]
                current_layer = (name+'_'+str(mark), module_dict[layer[0]](self.last_f))
            # 匹配批规范化
            elif linear_re.match(layer[0]) is not None:
                name = layer[0]
                current_layer = (name+'_'+str(mark), module_dict[layer[0]](*layer[1:]))
                self.last_f = layer[-1]
            # 匹配全连接
            else:
                name = layer[0]
                try:
                    if len(layer) > 1:
                        current_layer = (name+'_'+str(mark), module_dict[layer[0]](*layer[1:]))
                    # 带参数模块
                    else:
                        current_layer = (name+'_'+str(mark), module_dict[layer[0]])
                except KeyError:
                    current_layer = (name+'_'+str(mark), layer[1])
                # 非预设模块
            self.struct.add_module(*current_layer)
                

    def evaluate(self, test_data: DataLoader, mode: str = 'cls', details: bool = True):
        """评估模型

        - 参数:\n 
            test_data - 测试集\n 
            mode - 选定模型输出类型，'cls'为分类问题，'reg'为回归问题 - 待补充...\n
            details - 设置是否输出评估细节 - 待补充...
        """
        self.eval()
        res = 0
        add = 0
        if not self.test_num:
            add = 1
        with torch.no_grad():
            for data, labels in test_data:
                data, labels = data.to(self.device), labels.to(self.device)
                output = self.forward(data)
                if add:
                    self.test_num += len(data)
                if mode == 'cls':
                    res += ((output.argmax(dim=1)-labels) == 0).sum()
                    # 保存正确分类的个数
        if mode == 'cls':
            acry = res/self.test_num
            print('Accuracy - {:.2f} %'.format(acry*100))
        return acry


    def start_training(self, training_data: DataLoader, test_data: DataLoader, optimizer: Union[str, Optimizer], epochs: int,
                        lr: float, weight_decay: float, momentum: float = 0.9, eval: bool = True, figure: bool = False):
        """开始训练网络

        - 参数 :\n 
            training_data - 训练集\n 
            test_data - 测试集\n 
            optimizer - 训练使用的优化器, 可选参数'Adam', 'SGD'。抑或自行传入优化器为参数\n 
            epochs - 训练迭代次数\n
            lr - 学习率，推荐0~1\n
            weight_decay - 权重衰减的系数\n
            eval - 设置是否在每个epoch后测试模型\n 
            figure - 设置是否绘制训练集损失与测试集正确率\n 
        """
        self.train()
        # 设置训练模式, 在网络中含有BatchNorm或Dropout时有意义
        if type(optimizer) == str:
            if optimizer.lower() == 'adam':
                opt = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer.lower() == 'sgd':
                opt = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        # 预设优化器
        else:
            opt = optimizer
        # 自订优化器
        loss_list = []
        # 存储每个epoch的loss
        accuracy_list = []
        # 记录测试集的正确率 - 未启用
        begin = time.time()
        for epoch in range(epochs):
            epoch_time = time.time()
            for data, labels in training_data:
                data, labels = data.to(self.device), labels.to(self.device)
                # 将数据转移至指定设备 .to(torch.float32)
                opt.zero_grad()
                # 清空梯度
                output = self.forward(data)
                # .argmax(dim=1).to(torch.float32).requires_grad_(True)
                # 前向传播
                loss = self.loss_func(output, labels)
                loss.backward()
                # 反向传播
                opt.step()
                # 更新权值
            loss_list.append(loss)
            # 记录当前损失
            if not epoch or loss < min_loss:
                min_loss = loss
                torch.save(self.state_dict(), 'train_best.pt')
            # 保存训练集上最优模型
            print('Epoch {}/{} : training loss - {:.4f} time: {:.2f}s'.format(epoch+1, epochs, loss, time.time()-epoch_time))
            if eval:
                accuracy_list.append(self.evaluate(test_data=test_data, details=True))
                if not epoch or accuracy_list[-1] > max_acr:
                    max_acr = accuracy_list[-1]
                    torch.save(self.state_dict(), 'test_best.pt')
                # 保存测试集上最优模型
                self.train()
            # 测试
        total_time = [round(float(item)) for item in str(timedelta(seconds=time.time()-begin)).split(':')]
        print('Finished in {}h {}m {}s.'.format(*total_time))
            
        if figure:
            self._draw_figure(epochs, loss_list)
        # 绘制图像
        
    def _draw_figure(self, epochs: int, loss: list):
        "绘制图像并保存"
        plt.figure(figsize=(9, 9), dpi=200)
        x = np.array([i for i in range(epochs)])
        y = np.asarray(loss)
        plt.plot(x, y, color = 'blue')
        plt.title('Model Result')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('./Model Figure')

