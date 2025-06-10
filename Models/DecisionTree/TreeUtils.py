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


def cal_mse_(values):
    """计算均方误差"""
    return np.mean((values - np.mean(values)) ** 2)


def cal_weighted_mse_(values, weights):
    """计算加权均方误差"""
    return np.average((values - np.average(values, weights=weights)) ** 2, weights=weights)


def cal_mae_(values):
    """计算平均绝对误差"""
    return np.mean(np.abs(values - np.mean(values)))


def cal_weighted_mae_(values, weights):
    """计算加权平均绝对误差"""
    return np.average(np.abs(values - np.average(values, weights=weights)), weights=weights)


try:
    # 尝试导入numba
    import numba as nb
    from numba import jit


    @jit(nopython=True, cache=True)
    def cal_mse(values):
        """计算均方误差(jit加速)"""
        return np.mean((values - np.mean(values)) ** 2)


    @jit(nopython=True, cache=True)
    def weighted_mean(values, weights):
        """计算加权均值(jit加速)"""
        total_weight = 0.0
        weighted_sum = 0.0
        for i in range(len(values)):
            weighted_sum += values[i] * weights[i]
            total_weight += weights[i]
        return weighted_sum / total_weight


    @jit(nopython=True, cache=True)
    def cal_weighted_mse(values, weights):
        """计算加权均方误差(jit加速)"""
        mean = weighted_mean(values, weights)
        weighted_sum = 0.0
        total_weight = 0.0
        for i in range(len(values)):
            weighted_sum += weights[i] * (values[i] - mean) ** 2
            total_weight += weights[i]
        return weighted_sum / total_weight


    @jit(nopython=True, cache=True)
    def cal_mae(values):
        """计算平均绝对误差(jit加速)"""
        return np.mean(np.abs(values - np.mean(values)))


    @jit(nopython=True, cache=True)
    def cal_weighted_mae(values, weights):
        """计算加权平均绝对误差(jit加速)"""
        mean = weighted_mean(values, weights)
        weighted_sum = 0.0
        total_weight = 0.0
        for i in range(len(values)):
            weighted_sum += weights[i] * np.abs(values[i] - mean)
            total_weight += weights[i]
        return weighted_sum / total_weight

except ImportError:
    # 如果导入numba加速库失败，使用原始的函数
    warnings.warn("Without using numba acceleration...")
    cal_mse = cal_mse_
    cal_mae = cal_mae_
    cal_weighted_mse = cal_weighted_mse_
    cal_weighted_mae = cal_weighted_mae_

