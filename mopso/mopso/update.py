# -*- 编码格式：utf-8 -*-
# 作者：常冥
# 创建时间：2022/5/24
# encoding: utf-8
import numpy as np
import random
from mopso import pareto
from mopso import archiving


def update_v(v_, v_min, v_max, in_, in_pbest, in_gbest, w, c1, c2):
    # 更新速度值
    v_temp = w * v_ + c1 * (in_pbest - in_) + c2 * (in_gbest - in_)
    # 如果粒子的新速度大于最大值，则置为最大值；小于最小值，则置为最小值
    for i in range(v_temp.shape[0]):
        for j in range(v_temp.shape[1]):
            if v_temp[i, j, 1] < v_min[4]:
                v_temp[i, j, 1] = v_min[4]
            if v_temp[i, j, 1] > v_max[4]:
                v_temp[i, j, 1] = v_max[4]
    return v_temp


def update_in(in_, v_, in_range_list):
    # 更新位置参数
    in_temp = in_ + v_
    # 大于最大值，则置为最大值；小于最小值，则置为最小值
    for i in range(in_temp.shape[0]):
        for j in range(in_temp.shape[1]):
            # 层数变化
            if in_temp[i, j, 0] < in_range_list[j][0]:
                in_temp[i, j, 0] = in_range_list[j][0]
            if in_temp[i, j, 0] > in_range_list[j][1]:
                in_temp[i, j, 0] = in_range_list[j][1]
            # 增长率变化
            if in_temp[i, j, 1] < in_range_list[4][0]:
                in_temp[i, j, 1] = in_range_list[4][0]
            if in_temp[i, j, 1] > in_range_list[4][1]:
                in_temp[i, j, 1] = in_range_list[4][1]
    return in_temp


def compare_pbest(in_indiv, pbest_indiv):
    num_greater = 0
    num_less = 0
    for i in range(len(in_indiv)):
        if in_indiv[i] > pbest_indiv[i]:
            num_greater = num_greater + 1
        if in_indiv[i] < pbest_indiv[i]:
            num_less = num_less + 1
    # 如果当前粒子支配历史pbest，则更新,返回True
    if (num_greater > 0 and num_less == 0):
        return True
    # 如果历史pbest支配当前粒子，则不更新,返回False
    elif (num_greater == 0 and num_less > 0):
        return False
    else:
        # 如果互不支配，则按照概率决定是否更新
        random_ = random.uniform(0.0, 1.0)
        if random_ > 0.5:
            return True
        else:
            return False


def update_pbest(in_, fitness_, in_pbest, out_pbest):
    for i in range(out_pbest.shape[0]):
        # 通过比较历史pbest和当前粒子适应值，决定是否需要更新pbest的值。
        if compare_pbest(fitness_[i], out_pbest[i]):
            out_pbest[i] = fitness_[i]
            in_pbest[i] = in_[i]
    return in_pbest, out_pbest


def update_archive(in_, fitness_, archive_in, archive_fitness, thresh, mesh_div, range_list, particals):
    # 首先，计算当前粒子群的pareto边界，将边界粒子加入到存档archiving中
    pareto_1 = pareto.Pareto_(in_, fitness_)
    curr_in, curr_fit = pareto_1.pareto()
    # 其次，在存档中根据支配关系进行第二轮筛选，将非边界粒子去除
    in_new = np.concatenate((archive_in, curr_in), axis=0)
    fitness_new = np.concatenate((archive_fitness, curr_fit), axis=0)
    pareto_2 = pareto.Pareto_(in_new, fitness_new)
    curr_archiving_in, curr_archiving_fit = pareto_2.pareto()
    # 最后，判断存档数量是否超过了存档阀值。如果超过了阀值，则清除掉一部分（拥挤度高的粒子被清除的概率更大）
    if ((curr_archiving_in).shape[0] > thresh):
        clear_ = archiving.clear_archiving(curr_archiving_in, curr_archiving_fit, mesh_div, range_list, particals)
        curr_archiving_in, curr_archiving_fit = clear_.clear_(thresh)
    return curr_archiving_in, curr_archiving_fit


def update_gbest(archiving_in, archiving_fit, mesh_div, range_list, particals):
    get_g = archiving.get_gbest(archiving_in, archiving_fit, mesh_div, range_list, particals)
    return get_g.get_gbest()
