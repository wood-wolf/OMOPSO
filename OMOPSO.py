# -*- 编码格式：utf-8 -*-
# 作者：常冥
# 创建时间：2022/5/25
import numpy as np
from mopso import fitness_funs
from mopso import init
from mopso import update
from mopso import plot
from tqdm import tqdm


class Mopso(object):
    def __init__(self, particals, w, c1, c2, range_list, blocks, thresh, mesh_div=10):
        self.w, self.c1, self.c2 = w, c1, c2
        self.mesh_div = mesh_div
        self.particals = particals
        self.thresh = thresh
        self.range_list = range_list
        self.dim = blocks  # 块的个数等于维度
        self.max_v = []
        self.max_v = [1, 1, 1, 1, (range_list[4][1] - range_list[4][0]) * 0.05]  # 速度上限
        self.min_v = [1, 1, 1, 1, (range_list[4][1] - range_list[4][0]) * 0.05 * (-1)]  # 速度下限
        self.plot_ = plot.Plot_pareto()

    def initialize(self):
        # 初始化粒子
        self.in_ = init.init_designparams(self.particals, self.range_list, self.dim)
        # 初始化粒子速度
        self.v_ = init.init_v(self.particals, self.min_v, self.max_v, self.dim)
        # 计算适应值
        self.evaluation_fitness()
        # 初始化个体最优
        self.in_p, self.fitness_p = init.init_pbest(self.in_, self.fitness_)
        # 初始化外部存档
        self.archive_in, self.archive_fitness = init.init_archive(self.in_, self.fitness_)
        # 初始化全局最优
        self.in_g, self.fitness_g = init.init_gbest(self.archive_in, self.archive_fitness, self.mesh_div, self.range_list
                                                    , self.particals)

    def evaluation_fitness(self):
        # 计算适应值
        fitness_curr = []
        for i in range(self.in_.shape[0]):
            fitness_curr.append(fitness_funs.fitness(self.in_[i],self.dim))
        self.fitness_ = np.array(fitness_curr)  # 适应值

    def update_(self):
        # 更新粒子坐标、粒子速度、适应值、个体最优、外部存档、全局最优
        self.v_ = update.update_v(self.v_, self.min_v, self.max_v, self.in_, self.in_p, self.in_g, self.w, self.c1,
                                  self.c2)
        self.in_ = update.update_in(self.in_, self.v_, self.range_list)
        self.evaluation_fitness()
        self.in_p, self.fitness_p = update.update_pbest(self.in_, self.fitness_, self.in_p, self.fitness_p)
        self.archive_in, self.archive_fitness = update.update_archive(self.in_, self.fitness_, self.archive_in,
                                                                      self.archive_fitness, self.thresh, self.mesh_div,
                                                                      self.range_list, self.particals)
        self.in_g, self.fitness_g = update.update_gbest(self.archive_in, self.archive_fitness, self.mesh_div, range_list
                                                        , self.particals)

    def done(self, epochs):
        self.initialize()
        self.plot_.show(self.in_, self.fitness_, self.archive_in, self.archive_fitness, -1)
        for i in tqdm(range(epochs), desc='OMOPSO训练迭代中：'):
            self.update_()
            self.plot_.show(self.in_, self.fitness_, self.archive_in, self.archive_fitness, i)
        return self.archive_in, self.archive_fitness


if __name__ == '__main__':
    w = 0.8  # 惯性因子
    c1 = 0.1  # 局部速度因子
    c2 = 0.1  # 全局速度因子

    particals = 2  # 粒子群的数量
    epochs = 1  # 迭代次数

    mesh_div = 10  # 网格等分数量
    thresh = 300  # 外部存档阀值
    blocks = 4  # 粒子的维度（网络的块数量）
    range_list = [[4, 6], [4, 12], [4, 24], [4, 16], [8, 32]]
    mopso_ = Mopso(particals, w, c1, c2, range_list, blocks, thresh, mesh_div)  # 粒子群实例化
    pareto_in, pareto_fitness = mopso_.done(epochs)  # 经过epochs轮迭代后，pareto边界粒子
    print('怕累托粒子：',pareto_in)
    print('帕累托适应值：',pareto_fitness)
    np.savetxt("./img_txt/pareto_in.txt", pareto_in)  # 保存pareto边界粒子的坐标
    np.savetxt("./img_txt/pareto_fitness.txt", pareto_fitness)  # 打印pareto边界粒子的适应值
    # print("\n", "pareto边界的坐标保存于：./img_txt/pareto_in.txt")
    # print(" pareto边界的适应值保存于：./img_txt/pareto_fitness.txt")
    # print("\n", "迭代结束,over")
