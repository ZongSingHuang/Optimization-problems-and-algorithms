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

def transfer(V):
    # S-shaped (sigmoid)
    if V.ndim==1:
        V = V.reshape(1, -1)
    
    S = 1 / ( 1+np.exp(-V) )
    
    return S
    
#%% 參數設定
P = 60
D = 20
G = 1000
k = 0.2
w_max = 0.9
w_min = 0.2
c1 = 2
c2 = 2
ub = 1*np.ones([P, D])
lb = 0*np.ones([P, D])

#%% 初始化
X = np.random.choice(2, size=[P, D])
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
    
    # 轉換至[0, 1]
    S = transfer(V)
    
    # 更新X
    r3 = np.random.uniform(size=[P, D])
    X = 1.0*(S>r3)
    
#%% 作圖
plt.figure()
plt.plot(loss_curve)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Fitness')