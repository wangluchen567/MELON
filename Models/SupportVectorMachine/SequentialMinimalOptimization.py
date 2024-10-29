"""
序列最小优化算法
Sequential Minimal Optimization
"""
import numpy as np
from tqdm import tqdm


def smo_greedy_step(kernel_mat, x_train, y_train, alphas, b, C, tol=1.e-4):
    """
    SMO算法（贪婪运行一步）
    :param kernel_mat: 核函数矩阵
    :param x_train: 训练集数据
    :param y_train: 训练集标签
    :param alphas: 乘子参数
    :param b: 偏置参数
    :param C: 惩罚系数
    :param tol: 残差收敛条件（容忍系数）
    :return: alphas, b, optimize_end(已结束优化)
    """
    # 训练集形状
    num_data, num_dim = x_train.shape
    # 利用矩阵操作提前计算所有的误差E
    E_mat = kernel_mat @ (alphas * y_train) + b - y_train
    # 选择违反KKT条件最严重的一个元素
    condition1 = (y_train * E_mat < -tol) * (alphas < C)
    condition2 = (y_train * E_mat > tol) * (alphas > 0)
    # 得到所有元素违反约束的程度
    violation = np.zeros_like(alphas)
    violation[condition1] = C - alphas[condition1]
    violation[condition2] = alphas[condition2] - 0
    # 精度裁剪（以免在取最大值时出问题）
    violation = np.round(violation, decimals=9).flatten()
    # 检查是否没有可优化的项了
    if violation.sum() == 0:
        return alphas, b, True
    # 随机从违反最严重的元素中选择一个(若有多个的话)
    i = np.random.choice(np.arange(num_data)[violation == violation.max()])
    # 这里利用矩阵广播机制计算所有的差值
    diff_E = (float(E_mat[i]) - E_mat).flatten()
    # 这里利用矩阵操作提前计算所有alpha的上下界
    mask = np.array(y_train[i] != y_train).flatten()
    lowers, highers = np.zeros(num_data), np.zeros(num_data)
    lowers[mask] = np.max(np.hstack((np.zeros_like(y_train), alphas - alphas[i])), axis=1)[mask]
    highers[mask] = np.min(np.hstack((C + np.zeros_like(y_train), C + alphas - alphas[i])), axis=1)[mask]
    lowers[~mask] = np.max(np.hstack((np.zeros_like(y_train), alphas + alphas[i] - C)), axis=1)[~mask]
    highers[~mask] = np.min(np.hstack((C + np.zeros_like(y_train), alphas + alphas[i])), axis=1)[~mask]
    # 利用矩阵广播机制计算所有的eta值
    etas = kernel_mat[i, i] + np.diag(kernel_mat) - 2 * kernel_mat[i, :]
    # eta 类似于二阶导数值，只有当它大于0才能取最小值
    etas[etas <= 0] = 1.e-6  # 保证不除0
    # 利用矩阵计算所有更新后的alpha
    new_alphas = alphas + y_train * (diff_E / etas)[:, np.newaxis]
    # 第五步: 根据上下界对alpha裁剪
    new_alphas = np.max(np.hstack((new_alphas, lowers[:, np.newaxis])), axis=1)[:, np.newaxis]
    new_alphas = np.min(np.hstack((new_alphas, highers[:, np.newaxis])), axis=1)[:, np.newaxis]
    # 选择最大的下标
    """巨坑！这里涉及取第一个最大时存在的精度问题，这里设置为小数点后9位"""
    diff_alphas = np.round(np.abs(new_alphas - alphas), decimals=9).flatten()
    diff_alphas[i] = 0.0
    # 若没有更新则结束
    # 这里指定返回为True则是early stop
    if np.sum(diff_alphas) <= tol:
        return alphas, b, False
    # 若有多个最大值元素，则随机从所有的等于最大值元素中选择一个
    j = np.random.choice(np.arange(num_data)[diff_alphas == diff_alphas.max()])
    # 在更新alpha之前记录其旧值
    alpha_i_old, alpha_j_old = alphas[i].copy(), alphas[j].copy()
    # 第四步: 更新alpha_j
    alphas[j] = new_alphas[j]
    # 第六步：更新alpha_i
    alphas[i] = alpha_i_old + y_train[i] * y_train[j] * (alpha_j_old - alphas[j])
    """巨坑！这里涉及精度问题，这里设置为小数点后9位"""
    alphas = np.round(alphas, decimals=9)
    # 第七步：更新b_i和b_j
    b_i = float(b - float(E_mat[i]) - y_train[i] * (alphas[i] - alpha_i_old) * kernel_mat[i, i]
                - y_train[j] * (alphas[j] - alpha_j_old) * kernel_mat[j, i])
    b_j = float(b - float(E_mat[j]) - y_train[i] * (alphas[i] - alpha_i_old) * kernel_mat[i, j]
                - y_train[j] * (alphas[j] - alpha_j_old) * kernel_mat[j, j])
    # 第八步：根据b_i和b_j更新b
    if 0 < alphas[i] < C:
        b = b_i
    elif 0 < alphas[j] < C:
        b = b_j
    else:
        b = (b_i + b_j) / 2

    return alphas, b, False

def smo_greedy_step_regression(kernel_mat, x_train, y_train, alphas, b, C, epsilon, tol=1.e-4):
    """
    SMO算法（贪婪运行一步）
    :param kernel_mat: 核函数矩阵
    :param x_train: 训练集数据
    :param y_train: 训练集标签
    :param alphas: 乘子参数
    :param b: 偏置参数
    :param C: 惩罚系数
    :param epsilon: 损失容忍系数
    :param tol: 残差收敛条件（容忍系数）
    :return: alphas, b, optimize_end(已结束优化)
    """
    # 训练集形状
    num_data, num_dim = x_train.shape
    # 将alphas重整形状得到betas
    betas = np.hstack([alphas[:, 0], alphas[:, 1]])[:, np.newaxis]
    # 利用block得到矩阵Q_hat
    Q_mat = np.vstack((np.hstack((kernel_mat, kernel_mat)), np.hstack((kernel_mat, kernel_mat))))
    # 得到P矩阵和Z矩阵
    P_mat = np.vstack((epsilon + y_train, epsilon - y_train))
    Z_mat = np.vstack((np.ones((num_data, 1)), -np.ones((num_data, 1))))
    # 利用矩阵操作提前计算所有的输出情况
    """这里注意与之前不同,此处f(xi)为负的"""
    F_mat = - Q_mat @ (betas * Z_mat) + b
    # 然后得到所有误差值
    E_mat = Z_mat * P_mat - F_mat
    # 选择违反KKT条件最严重的一个元素
    condition1 = (betas < C) * (Z_mat * E_mat < -tol)
    condition2 = (betas > 0) * (Z_mat * E_mat > tol)
    # 还有一个约束是另一半是否是0，因为alpha和alpha*其中一个必为零
    condition3 = np.hstack([alphas[:, 1] != 0, alphas[:, 0] != 0])
    # 得到所有元素违反约束的程度
    violation = np.zeros_like(betas)
    violation[condition1] = C - betas[condition1]
    violation[condition2] = betas[condition2] - 0
    violation[condition3] = 0.0
    # 精度裁剪（以免在取最大值时出问题）
    violation = np.round(violation, decimals=9).flatten()
    # 检查是否没有可优化的项了
    if violation.sum() == 0:
        return alphas, b, True
    # 随机从违反最严重的元素中选择一个(若有多个的话)
    i = np.random.choice(np.arange(num_data * 2)[violation == violation.max()])
    # 需要利用矩阵广播机制计算beta情况下所有的差值
    diff_E = (float(E_mat[i]) - E_mat).flatten()
    # 这里利用矩阵操作提前计算所有beta的上下界
    mask = np.array(Z_mat[i] != Z_mat).flatten()
    lowers, highers = np.zeros(num_data * 2), np.zeros(num_data * 2)
    lowers[mask] = np.max(np.hstack((np.zeros_like(Z_mat), betas - betas[i])), axis=1)[mask]
    highers[mask] = np.min(np.hstack((C + np.zeros_like(Z_mat), C + betas - betas[i])), axis=1)[mask]
    lowers[~mask] = np.max(np.hstack((np.zeros_like(Z_mat), betas + betas[i] - C)), axis=1)[~mask]
    highers[~mask] = np.min(np.hstack((C + np.zeros_like(Z_mat), betas + betas[i])), axis=1)[~mask]
    # 利用矩阵广播机制计算所有的eta值
    etas = Q_mat[i, i] + np.diag(Q_mat) - 2 * Q_mat[i, :]
    # eta 类似于二阶导数值，只有当它大于0才能取最小值
    etas[etas <= 0] = 1.e-6  # 保证不除0
    # 利用矩阵计算所有更新后的betas
    new_betas = betas + Z_mat * (diff_E / etas)[:, np.newaxis]
    # 第五步: 根据上下界对betas裁剪
    new_betas = np.max(np.hstack((new_betas, lowers[:, np.newaxis])), axis=1)[:, np.newaxis]
    new_betas = np.min(np.hstack((new_betas, highers[:, np.newaxis])), axis=1)[:, np.newaxis]
    """巨坑！这里涉及取第一个最大时存在的精度问题，这里设置为小数点后9位"""
    diff_betas = np.round(np.abs(new_betas - betas), decimals=9).flatten()
    diff_betas[condition3] = 0.0
    diff_betas[i] = 0.0
    # 若没有更新则结束
    # 这里指定返回为True则是early stop
    if np.sum(diff_betas) <= tol:
        return alphas, b, False
    # 若有多个最大值元素，则随机从所有的等于最大值元素中选择一个
    j = np.random.choice(np.arange(num_data * 2)[diff_betas == diff_betas.max()])
    # 在更新beta之前记录其旧值
    beta_i_old, beta_j_old = betas[i].copy(), betas[j].copy()
    # 第四步: 更新beta_j
    betas[j] = new_betas[j]
    # 第六步：更新alpha_i
    betas[i] = beta_i_old + Z_mat[i] * Z_mat[j] * (beta_j_old - betas[j])
    """巨坑！这里涉及精度问题，这里设置为小数点后9位"""
    betas = np.round(betas, decimals=9)
    # 第七步：更新b_i和b_j
    b_i = float(b + float(E_mat[i]) + Z_mat[i] * (betas[i] - beta_i_old) * Q_mat[i, i]
                + Z_mat[j] * (betas[j] - beta_j_old) * Q_mat[j, i])
    b_j = float(b + float(E_mat[j]) + Z_mat[i] * (betas[i] - beta_i_old) * Q_mat[i, j]
                + Z_mat[j] * (betas[j] - beta_j_old) * Q_mat[j, j])
    # 第八步：根据b_i和b_j更新b
    if 0 < betas[i] < C:
        b = b_i
    elif 0 < betas[j] < C:
        b = b_j
    else:
        b = (b_i + b_j) / 2
    # 最后将betas转为alphas
    alphas[:, 0] = betas[:num_data].flatten()
    alphas[:, 1] = betas[num_data:].flatten()
    return alphas, b, False


def smo_random(x_train, y_train, C, tol, num_iter=1000):
    """序列最小优化算法(随机选择乘子)"""
    # 数据集大小
    num_data, num_dim = x_train.shape
    # 初始化参数（乘子alpha和b）
    alphas, b = np.zeros((num_data, 1)), 0
    # 迭代优化num_iter次
    for n in tqdm(range(num_iter)):
        # 遍历数据集
        for i in range(num_data):
            # 第一步：计算误差Ei
            f_xi = float((alphas * y_train).T @ (x_train @ x_train[i, np.newaxis].T)) + b
            E_i = f_xi - y_train[i]
            # 检查alpha是否需要被优化(即是否违反KKT条件)
            if ((y_train[i] * E_i < -tol) and (alphas[i] < C)) or ((y_train[i] * E_i > tol) and (alphas[i] > 0)):
                # 这里随机选择另一个需要与alpha_i成对优化的alpha_j
                cand_j = np.arange(num_data)
                cand_j = cand_j[cand_j != i]
                j = np.random.choice(cand_j)
                # 第一步：计算误差Ej
                f_xj = float((alphas * y_train).T @ (x_train @ x_train[j, np.newaxis].T))
                E_j = f_xj - y_train[j]
                # 在更新alpha之前记录其旧值
                alpha_i_old, alpha_j_old = alphas[i].copy(), alphas[j].copy()
                # 第二步：计算alpha的上下界
                if y_train[i] != y_train[j]:
                    lower = max(0, alphas[j] - alphas[i])
                    higher = min(C, C + alphas[j] - alphas[i])
                else:
                    lower = max(0, alphas[j] + alphas[i] - C)
                    higher = min(C, alphas[j] + alphas[i])
                # 若上下界相等则说明无法调整
                if lower == higher:
                    continue
                # 第三步：计算eta
                eta = float(x_train[i, np.newaxis] @ x_train[i, np.newaxis].T
                            + x_train[j, np.newaxis] @ x_train[j, np.newaxis].T
                            - 2 * x_train[i, np.newaxis] @ x_train[j, np.newaxis].T)
                # eta 类似于二阶导数值，只有当它大于0才能取最小值
                if eta <= 0:
                    continue
                # 第四步，更新alpha_j
                alphas[j] = alpha_j_old + y_train[j] * (E_i - E_j) / eta
                # 第五步：根据取值范围修剪alpha_j
                alphas[j] = lower if alphas[j] < lower else alphas[j]
                alphas[j] = higher if alphas[j] > higher else alphas[j]
                # 第六步：更新alpha_i
                alphas[i] = alpha_i_old + y_train[i] * y_train[j] * (alpha_j_old - alphas[j])
                # 第七步：更新b_i和b_j
                b_i = (b - E_i - y_train[i] * (alphas[i] - alpha_i_old) *
                       float(x_train[i, np.newaxis] @ x_train[i, np.newaxis].T)
                       - y_train[j] * (alphas[j] - alpha_j_old) *
                       float(x_train[j, np.newaxis] @ x_train[i, np.newaxis].T))
                b_j = (b - E_j - y_train[i] * (alphas[i] - alpha_i_old) *
                       float(x_train[i, np.newaxis] @ x_train[j, np.newaxis].T)
                       - y_train[j] * (alphas[j] - alpha_j_old) *
                       float(x_train[j, np.newaxis] @ x_train[j, np.newaxis].T))
                # 第八步：根据b_i和b_j更新b
                if 0 < alphas[i] < C:
                    b = b_i
                elif 0 < alphas[j] < C:
                    b = b_j
                else:
                    b = (b_i + b_j) / 2
    return alphas, b
