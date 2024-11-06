"""
决策树的节点类
"""
import numpy as np
from collections import deque


class ClassifierNode():
    """分类器节点"""
    def __init__(self, data, class_list):
        self.node_name = None  # 节点名称
        self.data = data  # 节点保存的数据集(最后一列为分类类别)
        self.class_types = data.iloc[:, -1]
        # 当前节点的类别(当前数据集中占比最多的类)
        self.category = self.get_most_freq(self.class_types)
        self.indicator = None  # 经过计算得到的指标值
        self.ind_type = None  # 指标类型('entropy'或'gini')
        self.samples = len(data)  # 当前类别数据集大小
        # 当前节点数据中每种类别数量
        self.values = [np.sum(self.class_types == class_) for class_ in class_list]
        self.state = None  # 节点分类状态
        self.pos = None  # 节点所在位置
        self.split_attr = None  # 节点要划分的属性
        self.split_value = None  # 连续问题要划分的值
        # 该节点的分支节点(若为空则为叶节点)
        self.branches = dict()

    def get_most_freq(self, array):
        # 获取唯一值及其计数
        unique, counts = np.unique(array, return_counts=True)
        # 找到最大计数的索引
        max_count_index = np.argmax(counts)
        # 找到出现次数最多的值
        most_freq_value = unique[max_count_index]
        return most_freq_value

    def assign_position(self):
        # 计算当前节点以及所有子节点位置的辅助函数
        queue = deque([(self, 0)])  # 队列初始化，包含根节点和其层级深度0
        # 保存每一层次的索引
        index_list = [0]
        depth = 0
        while queue:
            current_node, depth = queue.popleft()
            # 将x坐标设置为当前索引, 将y坐标设置为当前深度
            current_node.pos = (index_list[depth], depth)
            index_list[depth] += 1
            for branch in current_node.branches.values():
                queue.append((branch, depth + 1))  # 将子节点和其层级深度加入队列
                index_list.append(0)
        # 然后得到该树的最大深度和宽度
        self.max_depth = depth
        self.max_width = max(index_list)

class RegressorNode():
    def __init__(self, data):
        self.node_name = None  # 节点名称
        self.data = data  # 节点保存的数据集(最后一列为目标变量)
        self.target = data.iloc[:, -1]
        # 当前节点的预测值(当前数据集中目标变量的平均)
        self.predict_value = np.mean(self.target)
        self.indicator = None  # 经过计算得到的指标值
        self.ind_type = None  # 指标类型
        self.samples = len(data)  # 当前类别数据集大小
        self.state = None  # 节点划分状态
        self.pos = None  # 节点所在位置
        self.split_attr = None  # 节点要划分的属性
        self.split_value = None  # 节点划分时划分的值
        # 该节点的分支节点(若为空则为叶节点)
        self.branches = dict()

    def assign_position(self):
        # 计算当前节点以及所有子节点位置的辅助函数
        queue = deque([(self, 0)])  # 队列初始化，包含根节点和其层级深度0
        # 保存每一层次的索引
        index_list = [0]
        depth = 0
        while queue:
            current_node, depth = queue.popleft()
            # 将x坐标设置为当前索引, 将y坐标设置为当前深度
            current_node.pos = (index_list[depth], depth)
            index_list[depth] += 1
            for branch in current_node.branches.values():
                queue.append((branch, depth + 1))  # 将子节点和其层级深度加入队列
                index_list.append(0)
        # 然后得到该树的最大深度和宽度
        self.max_depth = depth
        self.max_width = max(index_list)