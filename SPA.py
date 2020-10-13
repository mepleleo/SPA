# !/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2020/9/20 
# @Author  : my_name_is_BUG
# @FileName: SPA.py
# @Software: PyCharm
# @Cnblogs ：https://blog.csdn.net/qq2512446791

# python: Python 3.7.3
# pandas: pandas 0.25.3
# numpy : numpy  1.16.2
# scipy : scipy  1.4.1
# progress:progress    1.5

import pandas as pd
import numpy as np
from scipy.linalg import qr, inv, pinv
import scipy.stats
import scipy.io as scio
from progress.bar import Bar
from matplotlib import pyplot as plt


class SPA:

    def _projections_qr(self, X, k, M):
        '''
        原版连续投影算法使用MATLAB内置的QR函数
        该版本改用scipy.linalg.qr函数
            https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.qr.html
        X : 预测变量矩阵
        K ：投影操作的初始列的索引
        M : 结果包含的变量个数

        return ：由投影操作生成的变量集的索引
        '''

        X_projected = X.copy()

        # 计算列向量的平方和
        norms = np.sum((X ** 2), axis=0)
        # 找到norms中数值最大列的平方和
        norm_max = np.amax(norms)

        # 缩放第K列 使其成为“最大的”列
        X_projected[:, k] = X_projected[:, k] * 2 * norm_max / norms[k]

        # 矩阵分割 ，order 为列交换索引
        _, __, order = qr(X_projected, 0, pivoting=True)

        return order[:M].T

    def _validation(self, Xcal, ycal, var_sel, Xval=None, yval=None):
        '''
        [yhat,e] = validation(Xcal,var_sel,ycal,Xval,yval) -->  使用单独的验证集进行验证
        [yhat,e] = validation(Xcal,ycalvar_sel) --> 交叉验证
        '''
        N = Xcal.shape[0]  # N 测试集的个数
        if Xval is None:  # 判断是否使用验证集
            NV = 0
        else:
            NV = Xval.shape[0]  # NV 验证集的个数

        yhat = e = None

        # 使用单独的验证集进行验证
        if NV > 0:
            Xcal_ones = np.hstack(
                [np.ones((N, 1)), Xcal[:, var_sel].reshape(N, -1)])

            # 对偏移量进行多元线性回归
            b = np.linalg.lstsq(Xcal_ones, ycal, rcond=None)[0]
            # 对验证集进行预测
            np_ones = np.ones((NV, 1))
            Xval_ = Xval[:, var_sel]
            X = np.hstack([np.ones((NV, 1)), Xval[:, var_sel]])
            yhat = X.dot(b)
            # 计算误差
            e = yval - yhat
        else:
            # 为yhat 设置适当大小
            yhat = np.zeros((N, 1))
            for i in range(N):
                # 从测试集中 去除掉第 i 项
                cal = np.hstack([np.arange(i), np.arange(i + 1, N)])
                X = Xcal[cal, var_sel.astype(np.int)]
                y = ycal[cal]
                xtest = Xcal[i, var_sel]
                # ytest = ycal[i]
                X_ones = np.hstack([np.ones((N - 1, 1)), X.reshape(N - 1, -1)])
                # 对偏移量进行多元线性回归
                b = np.linalg.lstsq(X_ones, y, rcond=None)[0]
                # 对验证集进行预测
                yhat[i] = np.hstack([np.ones(1), xtest]).dot(b)
            # 计算误差
            e = ycal - yhat

        return yhat, e

    def spa(self, Xcal, ycal, m_min=1, m_max=None, Xval=None, yval=None, autoscaling=1):
        '''
        [var_sel,var_sel_phase2] = spa(Xcal,ycal,m_min,m_max,Xval,yval,autoscaling) --> 使用单独的验证集进行验证
        [var_sel,var_sel_phase2] = spa(Xcal,ycal,m_min,m_max,autoscaling) --> 交叉验证

        如果 m_min 为空时， 默认 m_min = 1
        如果 m_max 为空时：
            1. 当使用单独的验证集进行验证时， m_max = min(N-1, K)
            2. 当使用交叉验证时，m_max = min(N-2, K)

        autoscaling : 是否使用自动刻度 yes = 1，no = 0, 默认为 1

        '''

        assert (autoscaling == 0 or autoscaling == 1), "请选择是否使用自动计算"

        N, K = Xcal.shape

        if m_max is None:
            if Xval is None:
                m_max = min(N - 1, K)
            else:
                m_max = min(N - 2, K)

        assert (m_max < min(N - 1, K)), "m_max 参数异常"

        # 第一步： 对测试集进行投影操作

        # 在均值中心化 和 自动窗口 之后 对 Xcal的列进行投影操作

        normalization_factor = None
        if autoscaling == 1:
            normalization_factor = np.std(
                Xcal, ddof=1, axis=0).reshape(1, -1)[0]
        else:
            normalization_factor = np.ones((1, K))[0]

        Xcaln = np.empty((N, K))
        for k in range(K):
            x = Xcal[:, k]
            Xcaln[:, k] = (x - np.mean(x)) / normalization_factor[k]

        SEL = np.zeros((m_max, K))

        # 进度条
        with Bar('Projections :', max=K) as bar:
            for k in range(K):
                SEL[:, k] = self._projections_qr(Xcaln, k, m_max)
                bar.next()

        # 第二步： 进行评估

        PRESS = float('inf') * np.ones((m_max + 1, K))

        with Bar('Evaluation of variable subsets :', max=(K) * (m_max - m_min + 1)) as bar:
            for k in range(K):
                for m in range(m_min, m_max + 1):
                    var_sel = SEL[:m, k].astype(np.int)
                    _, e = self._validation(Xcal, ycal, var_sel, Xval, yval)
                    PRESS[m, k] = e.T.dot(e)
                    bar.next()

        PRESSmin = np.min(PRESS, axis=0)
        m_sel = np.argmin(PRESS, axis=0)
        k_sel = np.argmin(PRESSmin)

        # 第 k_sel波段为初始波段时最佳，波段数目为 m_sel（k_sel）
        var_sel_phase2 = SEL[:m_sel[k_sel], k_sel].astype(np.int)

        # 最后消去变量

        # 第 3.1 步 计算相关指数
        Xcal2 = np.hstack([np.ones((N, 1)), Xcal[:, var_sel_phase2]])
        b = np.linalg.lstsq(Xcal2, ycal, rcond=None)[0]
        std_deviation = np.std(Xcal2, ddof=1, axis=0)

        relev = np.abs(b * std_deviation.T)
        relev = relev[1:]

        index_increasing_relev = np.argsort(relev, axis=0)
        index_decreasing_relev = index_increasing_relev[::-1].reshape(1, -1)[0]

        PRESS_scree = np.empty(len(var_sel_phase2))
        yhat = e = None
        for i in range(len(var_sel_phase2)):
            var_sel = var_sel_phase2[index_decreasing_relev[:i + 1]]
            _, e = self._validation(Xcal, ycal, var_sel, Xval, yval)

            PRESS_scree[i] = e.T.dot(e)

        RMSEP_scree = np.sqrt(PRESS_scree / len(e))

        # 第 3.3： F-test 验证
        PRESS_scree_min = np.min(PRESS_scree)
        alpha = 0.25
        dof = len(e)
        fcrit = scipy.stats.f.ppf(1 - alpha, dof, dof)
        PRESS_crit = PRESS_scree_min * fcrit

        # 找到不明显比 PRESS_scree_min 大的最小变量

        i_crit = np.min(np.nonzero(PRESS_scree < PRESS_crit))
        i_crit = max(m_min, i_crit)

        var_sel = var_sel_phase2[index_decreasing_relev[:i_crit]]

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        fig1 = plt.figure()
        plt.xlabel('Number of variables included in the model')
        plt.ylabel('RMSE')
        plt.title('Final number of selected variables:{}(RMSE={})'.format(len(var_sel), RMSEP_scree[i_crit]))
        plt.plot(RMSEP_scree)
        plt.scatter(i_crit, RMSEP_scree[i_crit], marker='s', color='r')
        plt.grid(True)

        fig2 = plt.figure()
        plt.plot(Xcal[0, :])
        plt.scatter(var_sel, Xcal[0, var_sel], marker='s', color='r')
        plt.legend(['First calibration object', 'Selected variables'])
        plt.xlabel('Variable index')
        plt.grid(True)
        plt.show()

        return var_sel, var_sel_phase2

    def __repr__(self):
        return "SPA()"


if __name__ == "__main__":
    data = pd.read_excel(r"F:\实验\data\all.xlsx")
    x = data.drop(['names', 'labels'], axis=1)
    y = data.loc[:, 'labels']

    absorbances = x.columns.values

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler


    Xcal, Xval, ycal, yval = train_test_split(x, y, test_size=0.4, random_state=0)

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))  # 这里feature_range根据需要自行设置，默认（0,1）

    Xcal = min_max_scaler.fit_transform(Xcal)
    Xval = min_max_scaler.transform(Xval)
   
    var_sel, var_sel_phase2 = SPA().spa(
        Xcal, ycal, m_min=2, m_max=50, Xval=Xval, yval=yval, autoscaling=1)
    print(absorbances[var_sel])
