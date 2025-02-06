class Model():
    def __init__(self, X_train, Y_train):
        self.X_train = X_train  # 训练数据
        self.Y_train = Y_train  # 真实标签

    def set_train_data(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def plot_2dim(self):
        raise NotImplementedError
