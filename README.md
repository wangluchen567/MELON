# 🍉 MELON
💬 Language：[English](README_EN.md) | [中文](README.md)

基于NumPy构建的机器学习算法与模型库<br>
Chen's MachinE Learning models Organized using Numpy

## 🔖 项目简介

这是一个基于NumPy实现的机器学习算法与模型库，旨在帮助对机器学习感兴趣的伙伴们深入理解机器学习各类算法和模型的底层原理。
为了降低学习门槛，本项目在设计上参考了scikit-learn的API风格，但主要聚焦于核心功能的实现，因此功能相对精简。与成熟的工业级框架不同，本项目更注重算法实现的透明性和可读性，通过手写底层代码来揭示机器学习模型的工作原理。
另外，需要说明的一点是，本项目更适合学习和实验用途，如需用于生产环境，这里推荐使用功能更完善的[scikit-learn](https://scikit-learn.org/)等专业框架。<br>
希望本项目能够帮助感兴趣的伙伴们建立起对机器学习模型与算法的深刻理解，为后续在机器学习与人工智能领域的进一步学习和研究打下坚实的基础。
<br><br>
**特别说明：`本代码仅供参考学习、竞赛和科学研究等非商业用途，在复制核心代码时请注明出处`**

## 📚 安装教程
**1. 建议使用 `Anaconda` 创建 `Python` 环境**

使用 Anaconda 创建环境可以方便地管理依赖包，避免版本冲突。建议从 [Anaconda 官网](https://www.anaconda.com/download/success) 下载并安装 Anaconda。如果需要特定版本，可以访问 [Anaconda所有版本下载地址](https://repo.anaconda.com/archive/)。

安装完成后，运行以下命令创建 Python 环境：
```bash
conda create --name my_env python=3.9
conda activate my_env
```
**注意**：本项目支持 Python 3.7 及以上版本，建议使用 Python 3.9 以获得最佳兼容性。

**2. 安装必要包**

本项目依赖以下包: `numpy`、`pandas`、`matplotlib`、`tqdm`、`networkx`。请确保已安装 Python 3.7 或更高版本，运行以下命令一键安装必要包：

```bash
pip install numpy pandas matplotlib tqdm networkx
```

**3. 镜像源选择**

如果在运行安装命令时发现下载速度较慢，可以尝试使用清华大学的镜像源进行安装。安装命令如下：
```bash
pip install numpy pandas matplotlib tqdm networkx -i https://pypi.tuna.tsinghua.edu.cn/simple
```
注意：如果无法访问上述镜像源，也可以选择其他可用的镜像源，例如中国科技大学、阿里云等。

## 🎯 核心实现

- **GradientOptimizer: 梯度优化器**
  - Optimizer: 包含GD/RMSProp/Adam等梯度优化方法 

- **SupportVectorMachine: 支持向量机**
  - SupportVectorClassifier: 支持向量机分类器
  - SupportVectorRegressor: 支持向量机分类器
  - SequentialMinimalOptimization: 序列最小化算法

- **DecisionTree: 决策树**
  - DecisionTreeClassifier: 决策树分类器
  - DecisionTreeRegressor: 决策树回归器
  - PlotTree: 决策树绘制

- **Clustering: 聚类算法**
  - KMeans: K-均值聚类
  - DBSCAN: 基于密度的空间聚类算法（带噪声处理）
  - SpectralClustering: 谱聚类

- **DimensionReduction: 降维相关模型**
  - PrincipalComponentAnalysis: PCA主成分分析

- **DiscriminantAnalysis: 判别分析相关模型**
  - FisherDiscriminantAnalysis: Fisher判别分析
  - GaussianDiscriminantAnalysis: 高斯判别分析
  - LinearDiscriminantAnalysis: 线性判别分析

- **LinearClassifier: 线性分类器**
  - GDClassifier: 使用梯度优化的分类器
  - LogisticRegression：逻辑回归分类器
  - Perceptron: 感知机分类器
  - RidgeClassifier: 岭回归分类器

- **LinearRegressor: 线性回归器**
  - GDRegressor: 使用梯度优化的回归器
  - LinearRegression: 最小二乘线性回归器
  - Ridge: 岭回归回归器

- **MultiClassWrapper: 多分类包装器**
  - OneVsOneClassifier: 一对一(OvO)分类包装器
  - OneVsRestClassifier: 一对多(OvR)分类包装器

- **NaiveBayes: 朴素贝叶斯相关模型**
  - GaussianNaiveBayes: 高斯朴素贝叶斯

- **NeighborsBased: 基于邻居相关模型**
  - KNeighborsClassifier: K近邻分类器
  - KNeighborsRegressor: K近邻回归器

- **EnsembleModels: 集成学习相关模型**
  - AdaBoostClassifier: 自适应提升分类器
  - AdaBoostRegressor: 自适应提升回归器
  - BaggingClassifier: 自助聚合分类器
  - BaggingRegressor: 自助聚合回归器
  - RandomForestClassifier: 随机森林分类器
  - RandomForestRegressor: 随机森林回归器
  - GradientBoostingClassifier: 梯度提升分类器
  - GradientBoostingRegressor： 梯度提升回归器

## 📦 项目结构

```
MELON/
├── Datasets/                           # 机器学习相关数据集
├── Examples/                           # 实现各种算法的具体实例
│   ├── CompareCluster.py               # 聚类算法的效果对比
│   ├── CompareLinearClassifier.py      # 线性分类器的效果对比
│   ├── CompareLinearRegressor.py       # 线性回归器的效果对比
│   ├── IrisClassifier.py               # 对鸢尾花数据集进行分类效果对比
│   ├── IrisClassifierBinary.py         # 对鸢尾花数据集进行二分类效果对比
├── Models/                             # 实现的机器学习模型
│   ├── Clustering/                     # 聚类算法相关模型
│   ├── DecisionTree/                   # 决策树相关模型
│   ├── DimensionReduction/             # 降维相关模型
│   ├── DiscriminantAnalysis/           # 判别分析相关模型
│   ├── EnsembleModels/                 # 集成学习相关模型
│   ├── GradientOptimizer/              # 梯度优化器
│   ├── LinearClassifier/               # 线性分类器
│   ├── LinearRegressor/                # 线性回归器
│   ├── MultiClassWrapper/              # 多分类包装器
│   ├── NaiveBayes/                     # 朴素贝叶斯相关模型
│   ├── NeighborsBased/                 # 基于邻居相关模型
│   ├── SupportVectorMachine/           # 支持向量机相关模型
│   ├── Model.py                        # 机器学习模型父类
│   └── Utils.py                        # 机器学习模型相关工具类
├── Notes/                              # 机器学习模型实现 参考笔记
└── README.md                           # 项目文档
```

## 📅 更新计划

- [x] 更新项目文档
- [ ] 更新算法笔记
- [x] 加入朴素贝叶斯分类器
- [x] 加入K-近邻算法
- [ ] 加入层次聚类算法
- [ ] 加入牛顿共轭梯度法
- [ ] 加入拟牛顿梯度法
- [x] 加入集成学习相关模型
- [x] 加入梯度提升相关模型
- [ ] 实现sample_weight参数
- [ ] debug AdaBoostClassifier for nonlinear classification


## 🌈 效果展示

### 梯度优化器

- 各类梯度优化算法优化平方函数的表现<br>
<img src="Notes/GradientOptimizer/Contrasts_1D.gif" width="288" height="220"/> <img src="Notes/GradientOptimizer/Contrasts_2D1.gif" width="288" height="220"/><br/>
- 各类梯度优化算法优化双曲抛物面(马鞍面)函数的表现<br>
<img src="Notes/GradientOptimizer/Contrasts_2D2.gif" width="288" height="220"/> <img src="Notes/GradientOptimizer/Contrasts_3D.gif" width="288" height="220"/><br/>

### 支持向量机
- **支持向量机分类器与SMO算法的原理与公式推导可参见笔记[支持向量机分类器](./Notes/SupportVectorMachine/SupportVectorClassifier.md)**
- 支持向量机分类器使用线性核函数对均匀随机数据和双点状随机数据分类效果<br>
<img src="Notes/SupportVectorMachine/SVC1.gif" width="288" height="220"/> <img src="Notes/SupportVectorMachine/SVC2.gif" width="288" height="220"/><br/>
- 支持向量机分类器使用不同伽马值的高斯核函数对同心圆随机数据分类效果<br>
<img src="Notes/SupportVectorMachine/SVC3.gif" width="288" height="220"/> <img src="Notes/SupportVectorMachine/SVC4.gif" width="288" height="220"/><br/>

- **支持向量机回归器与SMO算法的原理与公式推导可参见笔记[支持向量机回归器](./Notes/SupportVectorMachine/SupportVectorRegressor.md)**
- 支持向量机回归器使用线性核函数对随机线性数据回归效果/使用多项式核函数对随机多项式数据回归效果<br>
<img src="Notes/SupportVectorMachine/SVR1.gif" width="288" height="220"/> <img src="Notes/SupportVectorMachine/SVR2.gif" width="288" height="220"/><br/>
- 支持向量机回归器使用高斯核函数对三角函数数据回归效果/对复杂三角函数数据回归效果<br>
<img src="Notes/SupportVectorMachine/SVR3.gif" width="288" height="220"/> <img src="Notes/SupportVectorMachine/SVR4.gif" width="288" height="220"/><br/>

### 决策树

- 决策树分类器对均匀随机数据分类效果并绘制树的形状<br>
<img src="Notes/DecisionTree/DTC1.png" width="288" height="220"/> <img src="Notes/DecisionTree/DTC1Tree.png" width="288" height="220"/><br/>
- 决策树分类器对双点状随机数据和同心圆随机数据的分类效果<br>
<img src="Notes/DecisionTree/DTC2.png" width="288" height="220"/> <img src="Notes/DecisionTree/DTC3.png" width="288" height="220"/><br/>

- 决策树回归器对随机线性数据和随机多项式数据的回归效果<br>
<img src="Notes/DecisionTree/DTR1.png" width="288" height="220"/> <img src="Notes/DecisionTree/DTR2.png" width="288" height="220"/><br/>
- 决策树回归器对三角函数数据和对复杂三角函数数据回归效果<br>
<img src="Notes/DecisionTree/DTR3.png" width="288" height="220"/> <img src="Notes/DecisionTree/DTR4.png" width="288" height="220"/><br/>

### 线性分类器

- 使用梯度优化的线性分类器对均匀随机数据和双点状随机数据分类效果(默认hinge损失)<br>
<img src="Notes/LinearClassifier/GDC1.gif" width="288" height="220"/> <img src="Notes/LinearClassifier/GDC2.gif" width="288" height="220"/><br/>
- 逻辑回归线性分类器对均匀随机数据和双点状随机数据分类效果<br>
<img src="Notes/LinearClassifier/LogR1.gif" width="288" height="220"/> <img src="Notes/LinearClassifier/LogR2.gif" width="288" height="220"/><br/>
- 感知机分类器对均匀随机数据和双点状随机数据分类效果<br>
<img src="Notes/LinearClassifier/PP1.gif" width="288" height="220"/> <img src="Notes/LinearClassifier/PP2.gif" width="288" height="220"/><br/>

### 线性回归器
- 使用梯度优化的线性回归器和岭回归(Ridge)对线性数据回归效果<br>
<img src="Notes/LinearRegressor/GDR.gif" width="288" height="220"/> <img src="Notes/LinearRegressor/Ridge.png" width="288" height="220"/><br/>

### 聚类模型
- K-means、DBSCAN、谱聚类的聚类效果对比<br>
<img src="Notes/Clustering/Cluster.png" width="500" height="450"/><br/>

## 🤝 项目贡献

**Author: Luchen Wang**

## ✉️ 联系我们

**邮箱: wangluchen567@qq.com**

