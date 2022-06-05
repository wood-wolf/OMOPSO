# OMOPSO
这是Evolving Deep Neural Networks by Multi-objective Particle的复现；多目标优化粒子群算法+CNN网络；实现调参。
## pytorch 中Desnet.py修改
192行把 growth_rate -> growth_rate[i]
187行把 growth_rate -> growth_rate[i]
160行变成 growth_rate: [int,int,int,int] = [32,32,32,32],
## thop库 中 profile.py修改
173行变成 pass;#print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type)) 可以消除 [INFO]
176行变成 pass;#prRed("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type) 可以消除[WARN]
## 
