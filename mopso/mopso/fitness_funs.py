# -*- 编码格式：utf-8 -*-
# 作者：常冥
# 创建时间：2022/5/24
import numpy as np
from mopso import MOCNN


# 为了便于图示观察，试验测试函数为二维输入、二维输出
# 适应值函数：实际使用时请根据具体应用背景自定义 [[num_layers,growth]*4]
def fitness(in_, dim):
    growthRate = []
    blockConfig = []
    for i in range(dim):
        blockConfig.append(round(in_[i][0]))  # 把block的层数变成整数
        growthRate.append(int(round(in_[i][1])))  # 把 growthRate变成整数并且四舍五入取整
    # 输入增长率集合和4个块中每个块层数集合
    acc, flops = MOCNN.mocnn(growthRate, tuple(blockConfig))
    return [acc, flops]


def fitness_plot(in_):
    degree_45 = ((in_[0]-in_[1])**2/2)**0.5
    degree_135 = ((in_[0]+in_[1])**2/2)**0.5
    fit_1 = 1-np.exp(-(degree_45)**2/0.5)*np.exp(-(degree_135-np.sqrt(200))**2/250)
    fit_2 = 1-np.exp(-(degree_45)**2/5)*np.exp(-(degree_135)**2/350)
    return [fit_1,fit_2]
