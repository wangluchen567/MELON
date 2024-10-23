import warnings
import numpy as np
import pandas as pd
from PlotTree import plot_tree
from collections import deque


class Node():
    def __init__(self, data):
        self.data = data  # 数据集(最后一列为类型)
        # 当前节点的类别(当前数据集中占比最多的类)
        self.category = self.get_most_freq(data[:, -1])
        self.indicator = None  # 经过计算得到的指标值
        self.attr_name = None  # 分类的属性
        self.state = None
        self.pos = None  # 节点所在位置
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
        while queue:
            current_node, depth = queue.popleft()
            # 将x坐标设置为当前索引, 将y坐标设置为当前深度
            current_node.pos = (index_list[depth], depth)
            index_list[depth] += 1
            for branch in current_node.branches.values():
                queue.append((branch, depth + 1))  # 将子节点和其层级深度加入队列
                index_list.append(0)


class DecisionTreeClassifier():
    def __init__(self, X_train=None, Y_train=None, attr_names=None, attr_types=None, criterion='entropy',
                 splitter='best', max_depth=np.inf):
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.attr_names = None  # 属性名称
        self.attr_types = None  # 属性类别
        self.criterion = criterion  # 特征选择标准('entropy'或'gini')
        self.splitter = splitter  # 选择属性标准(贪婪或随机)
        self.max_depth = max_depth  # 决策树最大深度
        self.tree_depth = None  # 决策树的真实深度
        self.decision_tree = None  # 最终决策树
        self.set_train_data(X_train, Y_train)
        self.set_attributes(attr_names, attr_types)

    def set_train_data(self, X_train, Y_train):
        """给定训练数据集和标签数据"""
        if X_train is not None:
            if self.X_train is not None:
                warnings.warn("Training data will be overwritten")
            self.X_train = X_train.copy()
        if Y_train is not None:
            if self.Y_train is not None:
                warnings.warn("Training data will be overwritten")
            self.Y_train = Y_train.copy()

    def set_attributes(self, attr_names, attr_types):
        if attr_names is not None:
            if self.attr_names is not None:
                warnings.warn("attribute names will be overwritten")
            self.attr_names = attr_names.copy()
        else:
            if self.X_train is not None:
                # 属性名称默认为下标
                self.attr_names = list(range(self.X_train.shape[1]))
        if attr_types is not None:
            if self.attr_types is not None:
                warnings.warn("attribute types will be overwritten")
            self.attr_types = attr_types.copy()
        else:
            if self.X_train is not None:
                # 属性类别默认全部为离散值
                self.attr_types = np.zeros(self.X_train.shape[1])

    def train(self, X_train=None, Y_train=None, attr_names=None, attr_types=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        self.set_attributes(attr_names, attr_types)
        # 将数据和标签拼接方便决策树操作
        data = np.concatenate((self.X_train, self.Y_train), axis=1)
        self.decision_tree = Node(data)
        self.decision_tree.indicator = self.cal_attr_entropy(data)
        self.decision_tree.state = "root"
        # self.TreeGenerateRecursion(self.decision_tree, data, list(range(self.X_train.shape[1])))
        self.TreeGenerateQueue(self.decision_tree, data, list(range(self.X_train.shape[1])))
        print()

    def predict(self, X_data):
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")

    def TreeGenerateRecursion(self, node, data, attribute):
        """递归式生成决策树"""
        # 如果数据集均属于同一种类则将当前节点全都标记为该类叶节点
        if len(np.unique(data[:, -1])) == 1:
            return
        # 如果属性集为空或者数据集中样本在当前属性值集合上均属于同一种类
        # 则将当前节点标记为叶节点，且其类别标记为数据集中样本数最多的类
        if len(attribute) == 0:
            return
        # 从属性集中选择一个属性进行划分(贪婪选择或随机选择)
        attr = self.choice_attr(data, attribute)
        # 对当前属性中的每个值进行遍历
        # 若为离散值则得到当前属性中的所有类型
        uniques = np.unique(data[:, attr])
        # 遍历所有类型
        for t in uniques:
            # 为当前节点生成一个分支，得到在数据集中属性类型为t的样本子集
            data_t = data[data[:, attr] == t]
            # 产生新节点
            new_node = Node(data_t)
            node.attr_name = attr_names[attr]
            node.branches[t] = new_node
            new_node.state = f"{node.attr_name}={t}"
            # 计算信息熵值
            _, counts = np.unique(data_t[:, -1], return_counts=True)
            new_node.indicator = self.cal_entropy(counts)
            # 若样本子集为空，则将分支节点标记为叶节点，且其类别标记为数据集中样本数最多的类
            if len(data_t) == 0:
                return
            else:
                new_attr = attribute.copy()
                new_attr.remove(attr)
                self.TreeGenerateRecursion(new_node, data_t, new_attr)

    def TreeGenerateQueue(self, node, data, attribute):
        """队列式生成决策树"""
        # 初始化队列
        queue = deque([(node, data, attribute, 0)])
        # 遍历队列，插入节点
        while queue:
            # 从队列中弹出一组数据
            node, data, attribute, depth = queue.popleft()
            # 记录树的深度
            self.tree_depth = depth
            # 检查是否达到了指定的最大深度
            if depth >= self.max_depth:
                self.tree_depth = self.max_depth
                continue
            # 如果数据集均属于同一种类则将当前节点全都标记为该类叶节点
            if len(np.unique(data[:, -1])) == 1:
                continue
            # 如果属性集为空或者数据集中样本在当前属性值集合上均属于同一种类
            # 则将当前节点标记为叶节点，且其类别标记为数据集中样本数最多的类
            if len(attribute) == 0:
                continue
            # 从属性集中选择一个属性进行划分(贪婪选择或随机选择)
            attr = self.choice_attr(data, attribute)
            # 对当前属性中的每个值进行遍历
            # 若为离散值则得到当前属性中的所有类型
            uniques = np.unique(data[:, attr])
            # 遍历所有类型
            for t in uniques:
                # 为当前节点生成一个分支，得到在数据集中属性类型为t的样本子集
                data_t = data[data[:, attr] == t]
                # 产生新节点
                new_node = Node(data_t)
                node.attr_name = attr_names[attr]
                node.branches[t] = new_node
                new_node.state = f"{node.attr_name}={t}"
                # 计算信息熵值
                _, counts = np.unique(data_t[:, -1], return_counts=True)
                new_node.indicator = self.cal_entropy(counts)
                # 若样本子集为空，则将分支节点标记为叶节点，且其类别标记为数据集中样本数最多的类
                if len(data_t) == 0:
                    continue
                else:
                    new_attr = attribute.copy()
                    new_attr.remove(attr)
                    queue.append((new_node, data_t, new_attr, depth + 1))

    def choice_attr(self, data, attribute):
        """从当前属性集合中选择一个属性"""
        if self.splitter == 'random':
            # 随机选择
            attr = np.random.choice(attribute)
        elif self.splitter == 'best':
            # 贪婪选择
            # 遍历当前属性集合得到增益最大的属性
            gains = np.zeros(len(attribute))
            for i in range(len(gains)):
                gains[i] = self.cal_attr_entropy(data, index=attribute[i])
            attr = attribute[np.argmin(gains)]
        else:
            raise ValueError("There is no standard for selecting attributes like this")
        return attr

    def cal_attr_entropy(self, data, index=None):
        """给定数据集和属性下标计算信息熵值"""
        entropy = 0
        # 若属性下标指定为空则是计算根节点的信息熵
        if index is None:
            cate = data[:, -1]
            _, counts = np.unique(cate, return_counts=True)
            return self.cal_entropy(counts)
        # 获得当前属性列
        attr = data[:, index]
        # 得到当前属性列的情况
        attr_unique = np.unique(attr)
        # 遍历当前属性情况从而计算信息熵
        for a in attr_unique:
            cate = data[attr == a, -1]
            _, counts = np.unique(cate, return_counts=True)
            entropy += self.cal_entropy(counts) * len(cate) / len(attr)
        return entropy

    @staticmethod
    def cal_entropy(counts):
        counts = np.array(counts)
        probabilities = counts / counts.sum()
        # 为了保证是正的0这里加了0
        entropy = -np.sum(probabilities * np.log2(probabilities)) + 0.0
        return entropy


if __name__ == '__main__':
    model = DecisionTreeClassifier(max_depth=3)
    df = pd.read_csv("../../Dataset/watermelons2.csv")
    X_train = df.values[:, 1:-1]
    # Y_train = np.zeros((len(X_train), 1), dtype=int)
    # Y_train[df.values[:, -1] == '是'] = 1
    Y_train = df.values[:, -1].reshape(-1, 1)
    attr_names = df.columns[1:].tolist()
    model.train(X_train, Y_train, attr_names)
    model.decision_tree.assign_position()
    plot_tree(model.decision_tree)
