# -*- 编码格式：utf-8 -*-
# 作者：常冥
# 创建时间：2022/5/24
import numpy as np


def compare_(fitness_curr, fitness_ref):
    # 判断fitness_curr是否可以被fitness_ref完全支配
    for i in range(len(fitness_curr)):
        if fitness_curr[i] < fitness_ref[i]:
            return True
    return False


def judge_(fitness_curr, fitness_data, cursor):
    # 当前粒子的适应值fitness_curr与数据集fitness_data进行比较，判断是否为非劣解
    for i in range(len(fitness_data)):
        if i == cursor:
            continue
        # 如果数据集中存在一个粒子可以完全支配当前解，则证明当前解为劣解，返回False
        if not compare_(fitness_curr, fitness_data[i]):
            return False
    return True


class Pareto_(object):
    def __init__(self, in_data, fitness_data):
        self.in_data = in_data  # 粒子群坐标信息
        self.fitness_data = fitness_data  # 粒子群适应值信息
        self.cursor = -1  # 初始化游标位置
        self.len_ = in_data.shape[0]  # 粒子群的数量
        self.bad_num = 0  # 非优解的个数

    def next(self):
        # 将游标的位置前移一步，并返回所在检索位的粒子坐标、粒子适应值
        self.cursor = self.cursor + 1
        return self.in_data[self.cursor], self.fitness_data[self.cursor]

    def hasNext(self):
        # 判断是否已经检查完了所有粒子
        return self.len_ > self.cursor + 1 + self.bad_num

    def remove(self):
        # 将非优解从数据集删除，避免反复与其进行比较。
        self.fitness_data = np.delete(self.fitness_data, self.cursor, axis=0)
        self.in_data = np.delete(self.in_data, self.cursor, axis=0)
        # 游标回退一步
        self.cursor = self.cursor - 1
        # 非优解个数，加1
        self.bad_num = self.bad_num + 1

    def pareto(self):
        while self.hasNext():
            # 获取当前位置的粒子信息
            in_curr, fitness_curr = self.next()
            # 判断当前粒子是否pareto最优
            if not judge_(fitness_curr, self.fitness_data, self.cursor):
                self.remove()
        return self.in_data, self.fitness_data
