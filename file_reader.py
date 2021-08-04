import random
import os
import re

def files_exist(*file_names):
    """判断文件是否存在

    - 参数\n 
        file_names - 待查询的文件路径名\n

    - 返回:\n 
        若待查询文件路径名列表中有任一文件存在返回True，若均不存在返回Fasle
    """
    for file in file_names:
        if os.path.exists(file):
            print('{} does exist!'.format(file))
            return True
    return False

class DataFetcher(object):
    "标签-数据处理类"

    def __init__(self, data_file: str, labels_file: str, encoding: str = 'utf-8'):
        """
        - 参数:\n 
            data_file - 数据集的文件\n
            label_file - 对应的标签文件\n  
            encoding - 打开文件的编码方式
        """
        super().__init__()
        self.data_file = data_file
        self.labels_file = labels_file
        self.encoding = encoding
        # 存储
        with open(data_file, 'r', encoding=encoding) as f:
            self.data_lines = f.readlines()
        with open(labels_file, 'r', encoding=encoding) as f:
            self.labels_lines = f.readlines()
        # 读取数据标签文件


    def hold_out_sep(self, test_percent: float):
        """留出法划分训练集与测试集

        - 参数:\n
            test_percent - 测试集数量在数据集中的占比\n 
        """
        
        # 读取数据
        total_idx = len(self.labels_lines)
        idx_list = [i for i in range(total_idx)]
        random.shuffle(idx_list)
        # 打乱下标
        
        suffix = re.compile('\.[a-zA-Z]+$')
        out_file_names = []
        try:
            data_match = suffix.search(self.data_file).group()
            label_match = suffix.search(self.labels_file).group()
        except AttributeError:
            raise FileNotFoundError('Can\'t locate file\'s suffix!')
        for tt in ['_train', '_test']:
            out_file_names.append(self.data_file.replace(data_match, tt+'_data'+data_match))
            out_file_names.append(self.labels_file.replace(label_match, tt+'_labels'+label_match))
        # 存储输出文件名
        if files_exist(*out_file_names):
            print('The same name of output files already exist!')
            ans = input('Delete all existed files ? [Y/N] : ')
            if ans.lower() == 'n':
                raise FileExistsError('The same name of output files already exist!')
            elif ans.lower() == 'y':
                for file in out_file_names:
                    os.remove(file)
            else:
                raise StopIteration('You Pressed Wrong Keys.')
        # 检查输出的文件名是否存在
        print('Data is being divided...')
        for i in range(len(idx_list)):
            if i < int(len(idx_list)*test_percent):
                with open(out_file_names[2], 'a+', encoding=self.encoding) as test_f:
                    test_f.writelines(str(self.data_lines[idx_list[i]]).strip('[]').replace(',', ' '))
                # 追加数据集
                with open(out_file_names[3], 'a+') as test_label_f:
                    test_label_f.writelines(str(self.labels_lines[idx_list[i]]))
                # 追加标签
            # 划分为测试集
            else:
                with open(out_file_names[0], 'a+', encoding=self.encoding) as train_f:
                    train_f.writelines(str(self.data_lines[idx_list[i]]).strip('[]').replace(',', ' '))
                # 追加数据集
                with open(out_file_names[1], 'a+') as train_label_f:
                    train_label_f.writelines(str(self.labels_lines[idx_list[i]]))
                # 追加标签
            # 划分为训练集
        print('{} of data is divided into training dataset, {} is divided into test dataset.'.format(
            len(idx_list)-int(len(idx_list)*test_percent), int(len(idx_list)*test_percent)))
        try:
            file_suffix = re.compile(r'[\\/]+\w+\.\w+$')
            match = file_suffix.search(out_file_names[0]).group()
            dirc = out_file_names[0].replace(match, '')
            if dirc == '.':
                print('Dataset saved in current dirctory.')
            else:
                print('Dataset saved in {}.'.format(dirc))
        except AttributeError:
            print('Dataset saved in current dirctory.')
        