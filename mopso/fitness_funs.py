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
        growthRate.append(in_[i][1])
    # 输入增长率集合和4个块中每个块层数集合
    acc, flops = MOCNN.mocnn(growthRate, tuple(blockConfig))
    return [acc, flops]
