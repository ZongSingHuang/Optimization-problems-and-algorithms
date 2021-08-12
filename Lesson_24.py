# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:09:03 2021

@author: zongsing.huang
"""
# =============================================================================
# 最佳解的適應值為0.7071(MAX)/-0.7017(MIN)
# 將PSO轉為求解最大化問題的快速方法有:[a] 1/(1+F), [b] -1*F。前者因為適應值經過轉換所以不直觀，故建議採後者
# 方法[a]的分母有+1，是因為要防止分母變成0
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

def fitness(X, species='min'):
    # Sphere
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum(X, axis=1) / (1 + np.sum(X**2, axis=1))
    
    if species=='max':
        F = -1*F
        
    return F

#%% 參數設定
P = 300
D = 2
G = 500
k = 0.2
w_max = 0.9
w_min = 0.2
c1 = 2
c2 = 2
ub = 5*np.ones([P, D])
lb = -5*np.ones([P, D])
species = 'max'

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
    # 適應值計算
    F = fitness(X, species)
    
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
    
    if species=='max':
        loss_curve[g] = -1*loss_curve[g]
    
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