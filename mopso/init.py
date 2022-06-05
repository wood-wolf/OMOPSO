# -*- 编码格式：utf-8 -*-
# 作者：常冥
# 创建时间：2022/5/24
import random
import numpy as np
from mopso import archiving
from mopso import pareto
from tqdm import tqdm


# xx初始化粒子种群层数和增长率
def init_designparams(particals, range_list, dim=4):
    in_ = []  # P粒子群大小3D[2,4,2]结构
    for i in tqdm(range(particals), desc='CNNs粒子初始化中：'):
        ind = [[] for i in range(4)]
        for j in range(dim):
            layer_num = random.randint(range_list[j][0], range_list[j][1])  # 层数的变化
            growth_rate = random.randint(range_list[4][0], range_list[4][1])  # 增长率的变化
            ind[j] = [layer_num, growth_rate]
        in_.append(ind)
    return np.array(in_)


def init_v(particals, v_max, v_min, dim=4):
    in_v = []
    for i in tqdm(range(particals), desc='CNNs粒子速度初始化中：'):
        ind_v = [[] for i in range(4)]
        for j in range(dim):
            growth_rate = random.uniform(0, 1) * (v_max[4] - v_min[4]) + v_min[4]  # 增长率的速度变化
            ind_v[j] = [1, growth_rate]
        in_v.append(ind_v)
    return np.array(in_v)


def init_pbest(in_, fitness_):
    return in_, fitness_


def init_archive(in_, fitness_):
    pareto_c = pareto.Pareto_(in_, fitness_)  # pareto实例化
    curr_archiving_in, curr_archiving_fit = pareto_c.pareto()  #
    return curr_archiving_in, curr_archiving_fit


def init_gbest(curr_archiving_in, curr_archiving_fit, mesh_div, range_list, particals):
    get_g = archiving.get_gbest(curr_archiving_in, curr_archiving_fit, mesh_div, range_list, particals)
    return get_g.get_gbest()
