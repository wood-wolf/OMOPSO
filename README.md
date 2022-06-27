# OMOPSO
这是Evolving Deep Neural Networks by Multi-objective Particle的复现；多目标优化粒子群算法+CNN网络；实现调参。
智能控制理论与技术的小组作业
# 使用方法
OMOPSO.py文件是主函数文件可以直接运行，需要修改的参数有particals;generations;另外MOCNN.py中的epoch也要改。
## pytorch 中Desnet.py修改
192行把 growth_rate -> growth_rate[i]
187行把 growth_rate -> growth_rate[i]
160行变成 growth_rate: [int,int,int,int] = [32,32,32,32],
## thop库 中 profile.py修改
173行变成 pass;#print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type)) 可以消除 [INFO]
176行变成 pass;#prRed("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type) 可以消除[WARN]
## 注意事项：
第二个和第三个是为了符合论文作者中的编码规则进行修改的，因为我是调用了pytorch中的Desnet库进行操作的。你要是用其他的神经网络进行训练也是可行的。
