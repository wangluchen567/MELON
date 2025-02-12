"""
决策树回归器模型
Decision Tree Regressor
"""
import warnings
import numpy as np
import pandas as pd
from collections import deque
from Models.DecisionTree.PlotTree import plot_tree
from Models.DecisionTree.Node import RegressorNode
from Models.Utils import (cal_mse_metrics, run_uniform_regression,
                          plot_2dim_regression_sample, run_circular_regression, run_poly_regression)


class DecisionTreeRegressor():
    def __init__(self, X_train=None, Y_train=None, criterion='mse', splitter='best', max_depth=np.inf):
        """
        :param X_train: 训练数据
        :param Y_train: 真实目标值
        :param criterion: 特征划分标准(mse/mae)
        :param splitter: 选择属性标准(best/random)
        :param max_depth: 决策树最大深度
        """
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实目标值
        self.train_data = None  # 训练数据集（训练数据和真实目标值的整合）
        self.X_columns = None  # 训练数据的列名称
        self.attributes = None  # 特征名称
        self.criterion = criterion  # 特征划分标准(mse/mae)
        self.splitter = splitter  # 选择属性标准(best/random)
        self.max_depth = max_depth  # 决策树最大深度
        self.tree_depth = None  # 决策树的真实深度
        self.decision_tree = None  # 最终得到的决策树
        self.set_train_data(X_train, Y_train)

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

    def set_parameters(self, criterion=None, splitter=None, max_depth=None):
        """重新修改相关参数"""
        parameters = ['criterion', 'splitter', 'max_depth']
        values = [criterion, splitter, max_depth]
        for param, value in zip(parameters, values):
            if value is not None and getattr(self, param) is not None:
                warnings.warn(f"Parameter '{param}' will be overwritten")
                setattr(self, param, value)

    def train(self, X_train=None, Y_train=None, criterion=None, splitter=None, max_depth=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        self.set_parameters(criterion, splitter, max_depth)
        # 初始化决策树根节点
        self.decision_tree = RegressorNode(self.train_data)
        self.decision_tree.indicator = self.cal_indicator(self.train_data)
        self.decision_tree.ind_type = self.criterion
        self.decision_tree.state = "root"
        # self.TreeGenerateRecursion(self.decision_tree, self.train_data, self.attributes.copy())
        self.TreeGenerateQueue(self.decision_tree, self.train_data, self.attributes.copy())

    def predict(self, X_data_):
        X_data = X_data_.copy()
        # 若给定数据不是Dataframe或Series，则必须封装为Dataframe或Series才可以训练
        if not (isinstance(X_data, pd.DataFrame) or isinstance(X_data, pd.Series)):
            X_data = pd.DataFrame(X_data, columns=self.X_columns)
        # 决策树只能遍历得到每个数据的预测值
        Y_predict = []
        for i in range(len(X_data)):
            pointer = self.decision_tree
            while len(pointer.branches):
                # 检查是否是离散特征
                if self.check_discrete(self.train_data[pointer.split_attr].dtype):
                    pointer = pointer.branches[X_data.iloc[i][pointer.split_attr]]
                else:
                    if X_data.iloc[i][pointer.split_attr] <= pointer.split_value:
                        pointer = pointer.branches['True']
                    else:
                        pointer = pointer.branches['False']
            Y_predict.append(pointer.predict_value)
        Y_predict = np.array(Y_predict).reshape(-1, 1)
        return Y_predict

    def TreeGenerateRecursion(self, node, data, attributes):
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
                # 产生新节点
                new_node = RegressorNode(data_t)
                node.split_attr = attr
                node.node_name = f"{attr}=?"
                node.branches[t] = new_node
                new_node.state = f"{attr}={t}"
                # 计算指标值
                new_node.ind_type = self.criterion
                new_node.indicator = self.cal_indicator(data_t)
                # 若样本子集为空，则将分支节点标记为叶节点，且其类别标记为数据集中样本数最多的类
                if len(data_t) == 0:
                    return
                else:
                    new_attr = attributes.copy()
                    new_attr.remove(attr)
                    self.TreeGenerateRecursion(new_node, data_t, new_attr)
        # 若为连续值则拆分为两个分支
        else:
            # 需要遍历两种可能（不大于和大于）
            # 加入不大于的情况（左子节点）
            data_t = data[data[attr] <= divide]
            # 产生新节点
            new_node = RegressorNode(data_t)
            node.split_attr = attr
            node.split_value = divide
            node.node_name = f"{attr}<={divide:.3f}?"
            node.branches['True'] = new_node
            new_node.state = f"{attr}<={divide}"
            # 计算指标值
            new_node.ind_type = self.criterion
            new_node.indicator = self.cal_indicator(data_t)
            # 若样本子集为空，则将分支节点标记为叶节点，且其类别标记为数据集中样本数最多的类
            if len(data_t) == 0:
                return
            else:
                self.TreeGenerateRecursion(new_node, data_t, attributes.copy())
            # 加入大于的情况（右子节点）
            data_t = data[data[attr] > divide]
            # 产生新节点
            new_node = RegressorNode(data_t)
            node.split_attr = attr
            node.split_value = divide
            node.node_name = f"{attr}<={divide:.3f}?"
            node.branches['False'] = new_node
            new_node.state = f"{attr}>{divide}"
            # 计算指标值
            new_node.ind_type = self.criterion
            new_node.indicator = self.cal_indicator(data_t)
            # 若样本子集为空，则将分支节点标记为叶节点，且其类别标记为数据集中样本数最多的类
            if len(data_t) == 0:
                return
            else:
                self.TreeGenerateRecursion(new_node, data_t, attributes.copy())

    def TreeGenerateQueue(self, node, data, attributes):
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
                    new_node = RegressorNode(data_t)
                    node.split_attr = attr
                    node.node_name = f"{attr}=?"
                    node.branches[t] = new_node
                    new_node.state = f"{attr}={t}"
                    # 计算指标值
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
                new_node = RegressorNode(data_t)
                node.split_attr = attr
                node.split_value = divide
                node.node_name = f"{attr}<={divide:.3f}?"
                node.branches['True'] = new_node
                new_node.state = f"{attr}<={divide}"
                # 计算指标值
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
                new_node = RegressorNode(data_t)
                node.split_attr = attr
                node.split_value = divide
                node.node_name = f"{attr}<={divide:.3f}?"
                node.branches['False'] = new_node
                new_node.state = f"{attr}>{divide}"
                # 计算指标值
                new_node.ind_type = self.criterion
                new_node.indicator = self.cal_indicator(data_t)
                # 若样本子集为空，则将分支节点标记为叶节点，且其类别标记为数据集中样本数最多的类
                if len(data_t) == 0:
                    continue
                else:
                    queue.append((new_node, data_t, attributes.copy(), depth + 1))

    def choice_attr(self, data, attribute):
        """从当前属性集合中选择一个属性"""
        # 遍历当前属性集合得到误差情况
        errors = np.zeros(len(attribute))
        # 记录划分点情况（只有连续属性特征有效）
        divides = np.zeros_like(errors)
        # 得到所有属性集合的误差和划分点情况
        for i in range(len(errors)):
            errors[i], divides[i] = self.cal_attr_errors(data, attribute[i])
        # 根据误差和划分点选择属性
        if self.splitter == 'random':  # 随机选择
            index = np.random.randint(len(attribute))
        elif self.splitter == 'best':  # 贪婪选择
            index = np.argmin(errors)
        else:
            raise ValueError("There is no standard for selecting attributes like this")
        return attribute[index], divides[index]

    def cal_attr_errors(self, data, attr_key):
        """给定数据集和属性下标计算误差值"""
        # 若是离散特征
        if self.check_discrete(data[attr_key].dtype):
            return self.cal_error_discrete(data, attr_key), 0
        # 若是连续特征
        else:
            return self.cal_error_continuous(data, attr_key)

    def cal_error_discrete(self, data, attr_key):
        """计算误差值(离散特征)"""
        error_value = 0
        # 获得当前属性列
        attr_value = data[attr_key]
        # 得到当前属性列的情况
        attr_unique = np.unique(attr_value)
        # 遍历当前属性情况从而计算误差值
        for a in attr_unique:
            data_cal = data[attr_value == a]
            error_value += self.cal_indicator(data_cal) * len(data_cal) / len(attr_value)
        return error_value

    def cal_error_continuous(self, data, attr_key):
        """计算误差值(连续特征)"""
        # 获得当前属性列
        attr_value = data[attr_key]
        # 对数组进行排序
        sorted_attr = np.sort(attr_value)
        # 计算每两个数的中位点
        medians = (sorted_attr[:-1] + sorted_attr[1:]) / 2
        # 统计每个中位点划分的误差值
        errors = np.zeros_like(medians)
        # 遍历当前中位点属性情况从而计算误差值
        for i in range(len(medians)):
            # 先计算不大于该中位点的误差值
            data_cal = data[attr_value <= medians[i]]
            errors[i] += self.cal_indicator(data_cal) * len(data_cal) / len(attr_value)
            # 再计算大于该中位点的误差值
            data_cal = data[attr_value > medians[i]]
            errors[i] += self.cal_indicator(data_cal) * len(data_cal) / len(attr_value)
        # 选择误差值最小的为划分点，并返回划分点
        max_index = np.argmin(errors)
        error_value = errors[max_index]
        divide = medians[max_index]
        return error_value, divide

    def cal_indicator(self, data):
        target = data.iloc[:, -1]
        # 计算误差值
        if self.criterion == 'mse':
            return self.cal_mse(target)
        elif self.criterion == 'mae':
            return self.cal_mae(target)
        else:
            raise ValueError("There is no such indicator")

    @staticmethod
    def cal_mse(target):
        """计算均方误差"""
        return np.mean((target - np.mean(target)) ** 2)

    @staticmethod
    def cal_mae(target):
        """计算平均绝对误差"""
        return np.mean(np.abs(target - np.mean(target)))

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
        plot_tree(self.decision_tree)

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None):
        """为二维回归数据集和结果画图（只能是连续数据）"""
        X_train = np.array(self.X_train)
        Y_train = np.array(self.Y_train)
        plot_2dim_regression_sample(self, X_train, Y_train, X_test, Y_test)


if __name__ == '__main__':
    np.random.seed(100)
    model = DecisionTreeRegressor(max_depth=3, criterion='mse')
    run_uniform_regression(model)
    run_poly_regression(model)
    run_circular_regression(model)
    model.plot_tree()
