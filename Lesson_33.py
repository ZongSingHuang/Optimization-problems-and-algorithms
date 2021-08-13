# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:09:03 2021

@author: zongsing.huang
"""
# =============================================================================
# 最佳解為2500
# 題目X[4]和X[9]須為整數，故採用np.round進行修正
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
k = 0.2
w_max = 0.9
w_min = 0.2
c1 = 2
c2 = 2
ub = [10, 10, 10, 10, -50, 10, 10, 10, 10, 20]*np.ones([P, D])
lb = [-10, -10, -10, -10, -1000, -10, -10, -10, -10, -20]*np.ones([P, D])

#%% 初始化
X = np.random.uniform(low=lb, high=ub, size=[P, D])
V = np.zeros([P, D])
v_max = k*(ub-lb)*np.ones([P, D])
v_min = -1*v_max
pbest_X = np.zeros([P, D])
pbest_F = np.ones(P)*np.inf
gbest_X = np.zeros(D)
gbest_F = np.inf
loss_curve = np.zeros(G)

#%% 迭代
for g in range(G):
    # 修正X
    X[:, 4] = np.round(X[:, 4])
    X[:, -1] = np.round(X[:, -1])
    
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
    # 邊界處理
    mask1 = V>v_max
    mask2 = V<v_min
    V[mask1] = v_max[mask1]
    V[mask2] = v_min[mask2]
    
    # 更新X
    X = X + V
    # 邊界處理
    mask1 = X>ub
    mask2 = X<lb
    X[mask1] = ub[mask1]
    X[mask2] = lb[mask2]
    
#%% 作圖
plt.figure()
plt.plot(loss_curve)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Fitness')