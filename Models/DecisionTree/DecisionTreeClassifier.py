"""
Copyright (c) 2023 LuChen Wang
MELON is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
import warnings
import numpy as np
import pandas as pd
from Models import Model
from collections import deque
from Models.DecisionTree.PlotTree import plot_tree
from Models.DecisionTree.Node import ClassifierNode
from Models.Utils import (calculate_accuracy, run_uniform_classification, run_double_classification,
                          run_circle_classification, run_moons_classification, plot_2dim_classification_sample)


class DecisionTreeClassifier(Model):
    def __init__(self, X_train=None, Y_train=None, criterion='gini',
                 splitter='best', max_features=None, max_depth=np.inf):
        """
        决策树分类器模型
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param criterion: 特征选择标准(entropy/gini)
        :param splitter: 选择属性标准(best/random)
        :param max_features: 每次分裂时随机选择的最大特征数量(None/float/sqrt/log2)
        :param max_depth: 决策树最大深度
        """
        super().__init__(X_train, Y_train)
        self.train_data = None  # 训练数据集（训练数据和真实标签的整合）
        self.X_columns = None  # 训练数据的列名称
        self.attributes = None  # 特征名称
        self.criterion = criterion  # 特征选择标准(entropy/gini)
        self.splitter = splitter  # 选择属性标准(best/random)
        self.max_features = max_features  # 每次分裂时随机选择的最大特征数量
        self.max_depth = max_depth  # 决策树最大深度
        self.tree_depth = None  # 决策树的真实深度
        self.decision_tree = None  # 最终得到的决策树
        self.class_list = None  # 需要分类的类别情况
        self.sample_weight = None  # 样本的权重

    def set_train_data(self, X_train, Y_train):
        """给定训练数据集和标签数据"""
        if X_train is not None:
            if self.X_train is not None:
                warnings.warn("Training data will be overwritten")
            self.X_train = X_train.copy()
            # 若给定数据不是Dataframe或Series，则必须封装为Dataframe或Series才可以训练
            if not (isinstance(self.X_train, pd.DataFrame) or isinstance(self.X_train, pd.Series)):
                self.X_columns = ['dim_' + str(i + 1) for i in range(self.X_train.shape[1])]
                self.X_train = pd.DataFrame(self.X_train, columns=self.X_columns)
            else:
                self.X_columns = list(self.X_train.columns)
        if Y_train is not None:
            if self.Y_train is not None:
                warnings.warn("Training label will be overwritten")
            self.Y_train = Y_train.copy()
            # 若给定数据不是Dataframe或Series，则必须封装为Dataframe或Series才可以训练
            if not (isinstance(self.Y_train, pd.DataFrame) or isinstance(self.Y_train, pd.Series)):
                self.Y_train = pd.DataFrame(self.Y_train, columns=['label'])
        if X_train is not None and Y_train is not None:
            # 将两者整合成一个以方便训练
            self.train_data = pd.concat([self.X_train, self.Y_train], axis=1)
            self.attributes = self.train_data.columns[:-1].tolist()
            self.class_list = np.unique(self.train_data.iloc[:, -1])

    def train(self, X_train=None, Y_train=None, sample_weight=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        self.sample_weight = sample_weight
        # 初始化决策树根节点
        self.decision_tree = ClassifierNode(self.train_data, self.class_list)
        self.decision_tree.indicator = self.cal_indicator(self.train_data)
        self.decision_tree.ind_type = self.criterion
        self.decision_tree.state = "root"
        # self.build_tree_recursive(self.decision_tree, self.train_data, self.attributes.copy())
        self.build_tree_queue(self.decision_tree, self.train_data, self.attributes.copy())

    def predict(self, X_data_):
        """预测数据"""
        X_data = X_data_.copy()
        # 若给定数据不是Dataframe或Series，则必须封装为Dataframe或Series才可以预测
        if not (isinstance(X_data, pd.DataFrame) or isinstance(X_data, pd.Series)):
            X_data = pd.DataFrame(X_data, columns=self.X_columns)
        # 决策树只能遍历得到每个数据的分类类别
        Y_predict = []
        for i in range(len(X_data)):
            pointer = self.decision_tree
            while len(pointer.branches):
                # 检查是否是离散特征
                if self.check_discrete(self.train_data[pointer.split_attr].dtype):
                    # 获取当前数据的状态名称
                    state_name = X_data.iloc[i][pointer.split_attr]
                    # 若状态在所有分支中则继续检查其叶子
                    if state_name in pointer.branches.keys():
                        pointer = pointer.branches[state_name]
                    else:
                        # 否则选择默认分支(数据更多的一个分支)继续检查其叶子
                        pointer = pointer.branches[self.get_default(pointer.branches)]
                else:
                    if X_data.iloc[i][pointer.split_attr] <= pointer.split_value:
                        pointer = pointer.branches['True']
                    else:
                        pointer = pointer.branches['False']
            Y_predict.append(pointer.category)
        Y_predict = np.array(Y_predict).reshape(-1, 1)
        return Y_predict

    def predict_prob(self, X_data_):
        """预测数据(预测概率)"""
        X_data = X_data_.copy()
        # 若给定数据不是Dataframe或Series，则必须封装为Dataframe或Series才可以预测
        if not (isinstance(X_data, pd.DataFrame) or isinstance(X_data, pd.Series)):
            X_data = pd.DataFrame(X_data, columns=self.X_columns)
        # 初始化预测概率
        Y_predict_prob = np.zeros((len(X_data), len(self.class_list)))
        # 遍历所有数据得到每个数据的分类概率
        for i in range(len(X_data)):
            pointer = self.decision_tree
            while len(pointer.branches):
                # 检查是否是离散特征
                if self.check_discrete(self.train_data[pointer.split_attr].dtype):
                    # 获取当前数据的状态名称
                    state_name = X_data.iloc[i][pointer.split_attr]
                    # 若状态在所有分支中则继续检查其叶子
                    if state_name in pointer.branches.keys():
                        pointer = pointer.branches[state_name]
                    else:
                        # 否则选择默认分支(数据更多的一个分支)继续检查其叶子
                        pointer = pointer.branches[self.get_default(pointer.branches)]
                else:
                    if X_data.iloc[i][pointer.split_attr] <= pointer.split_value:
                        pointer = pointer.branches['True']
                    else:
                        pointer = pointer.branches['False']
            Y_predict_prob[i] = np.array(pointer.values) / np.sum(pointer.values)
        return Y_predict_prob

    def get_default(self, branches):
        """获取默认分支"""
        return max(branches, key=lambda k: len(branches[k].data))

    def build_tree_recursive(self, node, data, attributes):
        """递归式生成决策树(暂时无法控制深度)"""
        # 如果数据集均属于同一种类则将当前节点全都标记为该类叶节点
        if len(np.unique(data.iloc[:, -1])) == 1:
            return
        # 如果属性集为空或者数据集中样本在当前属性值集合上均属于同一种类
        # 则将当前节点标记为叶节点，且其类别标记为数据集中样本数最多的类
        if len(attributes) == 0:
            return
        # 从属性集中选择一个属性进行划分(贪婪选择或随机选择)
        attr, divide = self.choice_attr(data, attributes)
        # 对当前属性中的每个值进行遍历
        # 若为离散值则要遍历当前属性中的所有类型
        if self.check_discrete(data[attr].dtype):
            uniques = np.unique(data[attr])
            # 遍历所有类型
            for t in uniques:
                # 为当前节点生成一个分支，得到在数据集中属性类型为t的样本子集
                data_t = data[data[attr] == t]
                # 若新节点数据为空则使用父类数据(其实说明是叶节点)
                if len(data_t) == 0:
                    data_t = data.copy()
                # 产生新节点
                new_node = ClassifierNode(data_t, self.class_list)
                node.split_attr = attr
                node.node_name = f"{attr}=?"
                node.branches[t] = new_node
                new_node.state = f"{attr}={t}"
                # 计算指标值（信息熵或基尼指数）
                new_node.ind_type = self.criterion
                new_node.indicator = self.cal_indicator(data_t)
                # 若样本子集为空，则将分支节点标记为叶节点，且其类别标记为数据集中样本数最多的类
                if len(data_t) == 0:
                    return
                else:
                    new_attr = attributes.copy()
                    new_attr.remove(attr)
                    self.build_tree_recursive(new_node, data_t, new_attr)
        # 若为连续值则拆分为两个分支
        else:
            # 需要遍历两种可能（不大于和大于）
            # 加入不大于的情况（左子节点）
            data_t = data[data[attr] <= divide]
            # 产生新节点
            # 若新节点数据为空则使用父类数据(其实说明是叶节点)
            if len(data_t) == 0:
                data_t = data.copy()
            new_node = ClassifierNode(data_t, self.class_list)
            node.split_attr = attr
            node.split_value = divide
            node.node_name = f"{attr}<={divide:.3f}?"
            node.branches['True'] = new_node
            new_node.state = f"{attr}<={divide}"
            # 计算指标值（信息熵或基尼指数）
            new_node.ind_type = self.criterion
            new_node.indicator = self.cal_indicator(data_t)
            # 若样本子集为空，则将分支节点标记为叶节点，且其类别标记为数据集中样本数最多的类
            if len(data_t) == 0:
                return
            else:
                self.build_tree_recursive(new_node, data_t, attributes.copy())
            # 加入大于的情况（右子节点）
            data_t = data[data[attr] > divide]
            # 产生新节点
            # 若新节点数据为空则使用父类数据(其实说明是叶节点)
            if len(data_t) == 0:
                data_t = data.copy()
            new_node = ClassifierNode(data_t, self.class_list)
            node.split_attr = attr
            node.split_value = divide
            node.node_name = f"{attr}<={divide:.3f}?"
            node.branches['False'] = new_node
            new_node.state = f"{attr}>{divide}"
            # 计算指标值（信息熵或基尼指数）
            new_node.ind_type = self.criterion
            new_node.indicator = self.cal_indicator(data_t)
            # 若样本子集为空，则将分支节点标记为叶节点，且其类别标记为数据集中样本数最多的类
            if len(data_t) == 0:
                return
            else:
                self.build_tree_recursive(new_node, data_t, attributes.copy())

    def build_tree_queue(self, node, data, attributes):
        """队列式生成决策树"""
        # 初始化队列
        queue = deque([(node, data, attributes, 0)])
        # 遍历队列，插入节点
        while queue:
            # 从队列中弹出一组数据
            node, data, attributes, depth = queue.popleft()
            # 记录树的深度
            self.tree_depth = depth
            # 检查是否达到了指定的最大深度
            if depth >= self.max_depth:
                self.tree_depth = self.max_depth
                continue
            # 如果数据集均属于同一种类则将当前节点全都标记为该类叶节点
            if len(np.unique(data.iloc[:, -1])) == 1:
                continue
            # 如果属性集为空或者数据集中样本在当前属性值集合上均属于同一种类
            # 则将当前节点标记为叶节点，且其类别标记为数据集中样本数最多的类
            if len(attributes) == 0:
                continue
            # 从属性集中选择一个属性进行划分(贪婪选择或随机选择)
            attr, divide = self.choice_attr(data, attributes)
            # 对当前属性中的每个值进行遍历
            # 若为离散值则要遍历当前属性中的所有类型
            if self.check_discrete(data[attr].dtype):
                uniques = np.unique(data[attr])
                # 遍历所有类型
                for t in uniques:
                    # 为当前节点生成一个分支，得到在数据集中属性类型为t的样本子集
                    data_t = data[data[attr] == t]
                    # 产生新节点
                    # 若新节点数据为空则使用父类数据(其实说明是叶节点)
                    if len(data_t) == 0:
                        data_t = data.copy()
                    new_node = ClassifierNode(data_t, self.class_list)
                    node.split_attr = attr
                    node.node_name = f"{attr}=?"
                    node.branches[t] = new_node
                    new_node.state = f"{attr}={t}"
                    # 计算指标值（信息熵或基尼指数）
                    new_node.ind_type = self.criterion
                    new_node.indicator = self.cal_indicator(data_t)
                    # 若样本子集为空，则将分支节点标记为叶节点，且其类别标记为数据集中样本数最多的类
                    if len(data_t) == 0:
                        continue
                    else:
                        new_attr = attributes.copy()
                        new_attr.remove(attr)
                        queue.append((new_node, data_t, new_attr, depth + 1))
            # 若为连续值则拆分为两个分支
            else:
                # 需要遍历两种可能（不大于和大于）
                # 加入不大于的情况（左子节点）
                data_t = data[data[attr] <= divide]
                # 产生新节点
                # 若新节点数据为空则使用父类数据(其实说明是叶节点)
                if len(data_t) == 0:
                    data_t = data.copy()
                new_node = ClassifierNode(data_t, self.class_list)
                node.split_attr = attr
                node.split_value = divide
                node.node_name = f"{attr}<={divide:.3f}?"
                node.branches['True'] = new_node
                new_node.state = f"{attr}<={divide}"
                # 计算指标值（信息熵或基尼指数）
                new_node.ind_type = self.criterion
                new_node.indicator = self.cal_indicator(data_t)
                # 若样本子集为空，则将分支节点标记为叶节点，且其类别标记为数据集中样本数最多的类
                if len(data_t) == 0:
                    continue
                else:
                    queue.append((new_node, data_t, attributes.copy(), depth + 1))
                # 加入大于的情况（右子节点）
                data_t = data[data[attr] > divide]
                # 产生新节点
                # 若新节点数据为空则使用父类数据(其实说明是叶节点)
                if len(data_t) == 0:
                    data_t = data.copy()
                new_node = ClassifierNode(data_t, self.class_list)
                node.split_attr = attr
                node.split_value = divide
                node.node_name = f"{attr}<={divide:.3f}?"
                node.branches['False'] = new_node
                new_node.state = f"{attr}>{divide}"
                # 计算指标值（信息熵或基尼指数）
                new_node.ind_type = self.criterion
                new_node.indicator = self.cal_indicator(data_t)
                # 若样本子集为空，则将分支节点标记为叶节点，且其类别标记为数据集中样本数最多的类
                if len(data_t) == 0:
                    continue
                else:
                    queue.append((new_node, data_t, attributes.copy(), depth + 1))

    def choice_attr(self, data, attribute_):
        """从当前属性集合中选择一个属性"""
        if self.max_features is not None:
            num_feats = len(attribute_)
            # 若指定了随机的最大数量则随机选择属性
            if isinstance(self.max_features, float):
                # 给定的最大数量是浮点比例
                num_feats *= self.max_features
            elif self.max_features == 'sqrt':
                # 给定的最大数量是平方根比例
                num_feats = np.sqrt(len(attribute_))
            elif self.max_features == 'log2':
                # 给定的最大数量是log2比例
                num_feats = np.log2(len(attribute_))
            # 从属性集合中随机选择一定比例的属性
            attribute = np.random.choice(attribute_, size=int(num_feats), replace=False)
        else:
            # 若不指定随机选择则默认全部选择
            attribute = attribute_
        # 遍历当前属性集合得到增益情况
        gains = np.zeros(len(attribute))
        # 记录划分点情况（只有连续属性特征有效）
        divides = np.zeros_like(gains)
        # 得到所有属性集合的增益和划分点情况
        for i in range(len(gains)):
            gains[i], divides[i] = self.cal_attr_gains(data, attribute[i])
        # 根据增益和划分点选择属性
        if self.splitter == 'random':  # 随机选择
            index = np.random.randint(len(attribute))
        elif self.splitter == 'best':  # 贪婪选择
            index = np.argmax(gains)
        else:
            raise ValueError("There is no standard for selecting attributes like this")
        return attribute[index], divides[index]

    def cal_attr_gains(self, data, attr_key):
        """给定数据集和属性下标计算信息熵值"""
        # 若是离散特征
        if self.check_discrete(data[attr_key].dtype):
            return self.cal_gain_discrete(data, attr_key), 0
        # 若是连续特征
        else:
            return self.cal_gain_continuous(data, attr_key)

    def cal_gain_discrete(self, data, attr_key):
        """计算信息增益/基尼指数(离散特征)"""
        # 计算当前样本集合的不纯度作为初始值
        gain = self.cal_indicator(data)
        # 获得当前属性列
        attr_value = data[attr_key]
        # 得到当前属性列的情况
        attr_unique = np.unique(attr_value)
        # 遍历当前属性情况从而计算信息增益
        for a in attr_unique:
            data_cal = data[attr_value == a]
            # 子集在整个数据集中的权重比例(若无样本权重则为个数比值)
            if self.sample_weight is None:
                prop = len(data_cal) / len(attr_value)
            else:
                prop = (self.sample_weight[np.array(data_cal.index)].sum()
                        / self.sample_weight[np.array(attr_value.index)].sum())
            gain -= self.cal_indicator(data_cal) * prop
        return gain

    def cal_gain_continuous(self, data, attr_key):
        """计算信息增益/基尼指数(连续特征)"""
        # 获得当前属性列
        attr_value = data[attr_key]
        # 对数组进行去重并排序
        sorted_attr = np.unique(attr_value)
        # 若去重后只有一种数据则直接返回
        if len(sorted_attr) == 1:
            return 0, len(data)
        # 计算每两个数的中位点
        medians = (sorted_attr[:-1] + sorted_attr[1:]) / 2
        # 计算当前样本集合的不纯度作为初始值
        gains = np.zeros_like(medians) + self.cal_indicator(data)
        # 遍历当前中位点属性情况从而计算信息增益
        for i in range(len(medians)):
            # 先计算不大于该中位点的信息增益
            data_cal = data[attr_value <= medians[i]]
            # 子集在整个数据集中的权重比例(若无样本权重则为个数比值)
            if self.sample_weight is None:
                prop_left = len(data_cal) / len(attr_value)
            else:
                prop_left = (self.sample_weight[np.array(data_cal.index)].sum()
                             / self.sample_weight[np.array(attr_value.index)].sum())
            gains[i] -= self.cal_indicator(data_cal) * prop_left
            # 再计算大于该中位点的信息增益
            data_cal = data[attr_value > medians[i]]
            # 子集在整个数据集中的权重比例(若无样本权重则为个数比值)
            prop_right = 1 - prop_left  # 另一半的比例无需重复计算
            gains[i] -= self.cal_indicator(data_cal) * prop_right
        # 选择增益最大的为划分点，并返回划分点
        max_index = np.argmax(gains)
        gain = gains[max_index]
        divide = medians[max_index]
        return gain, divide

    def cal_indicator(self, data):
        cate = data.iloc[:, -1]
        if self.sample_weight is None:
            _, counts = np.unique(cate, return_counts=True)
        else:  # 若使用样本采样权重
            # 得到每个样本对应类别的索引
            _, inverse_indices = np.unique(cate, return_inverse=True)
            # 根据类别索引和样本权重计算每个类别的加权样本数量
            counts = np.bincount(inverse_indices, weights=self.sample_weight[np.array(data.index)])
        # 计算指标值(信息熵值或Gini指数值)
        if self.criterion == 'entropy':
            return self.cal_entropy(counts)
        elif self.criterion == 'gini':
            return self.cal_gini(counts)
        else:
            raise ValueError("There is no such indicator")

    @staticmethod
    def cal_entropy(counts):
        """计算信息熵值"""
        counts = np.array(counts)
        probabilities = counts / counts.sum()
        # 为了保证是正的0这里加了0
        entropy = -np.sum(probabilities * np.log2(probabilities)) + 0.0
        return entropy

    @staticmethod
    def cal_gini(counts):
        """计算基尼值"""
        counts = np.array(counts)
        probabilities = counts / counts.sum()
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    @staticmethod
    def check_discrete(dtype_):
        """检查是否是离散特征"""
        if dtype_ in [np.dtype('float16'), np.dtype('float32'), np.dtype('float64')]:
            return False
        else:
            return True

    def plot_tree(self):
        """绘制决策树"""
        self.decision_tree.assign_position()
        plot_tree(self.decision_tree, class_names=np.unique(self.train_data.iloc[:, -1]))

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None):
        """为二维分类数据集和结果画图（只能是连续数据）"""
        X_train = np.array(self.X_train)
        Y_train = np.array(self.Y_train)
        plot_2dim_classification_sample(self, X_train, Y_train, X_test, Y_test, neg_label=-1)


def run_watermelon_example():
    model = DecisionTreeClassifier(max_depth=3)
    data = pd.read_csv("../../Dataset/watermelons2.csv")
    X_train = data.iloc[:, 1:-1]
    Y_train = data.iloc[:, -1]
    model.train(X_train, Y_train)
    Y_predict = model.predict(X_train)
    print("准确率为: {:.3f} %".format(calculate_accuracy(Y_train, Y_predict) * 100))
    model.plot_tree()


if __name__ == '__main__':
    run_watermelon_example()
    np.random.seed(100)
    model = DecisionTreeClassifier(max_depth=5, criterion='gini')
    run_uniform_classification(model)
    run_double_classification(model)
    run_circle_classification(model)
    run_moons_classification(model)
    model.plot_tree()
