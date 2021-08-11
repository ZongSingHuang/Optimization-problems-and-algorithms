# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:09:03 2021

@author: zongsing.huang
"""
# =============================================================================
# 最佳解的適應值為0
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

def fitness(X):
    # Sphere
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(X**2, axis=1)
    
    return F

#%% 參數設定
P = 300
D = 10
G = 500
w_max = 0.9
w_min = 0.2
c1 = 2
c2 = 2
ub = 10*np.ones(D)
lb = -10*np.ones(D)

#%% 初始化
X = np.random.uniform(low=lb, high=ub, size=[P, D])
V = np.zeros([P, D])
pbest_X = np.zeros([P, D])
pbest_F = np.ones(P)*np.inf
gbest_X = np.zeros(D)
gbest_F = np.inf
loss_curve = np.zeros(G)

#%% 迭代
for g in range(G):
    # 適應值計算
    F = fitness(X)
    
    # 更新pbest
    mask = F<pbest_F
    pbest_X[mask] = X[mask]
    pbest_F[mask] = F[mask]
    
    # 更新gbest
    if F.min()<gbest_F:
        gbest_idx = np.argmin(F)
        gbest_X = X[gbest_idx]
        gbest_F = F[gbest_idx]
    loss_curve[g] = gbest_F
    
    # 更新w
    w = w_max - g*(w_max-w_min)/G
    
    # 更新V
    r1 = np.random.uniform(size=[P, D])
    r2 = np.random.uniform(size=[P, D])
    V = w*V + c1*r1*(pbest_X-X) + c2*r2*(gbest_X-X)
    
    # 更新X
    X = X + V
    
#%% 作圖
plt.figure()
plt.plot(loss_curve)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Fitness')