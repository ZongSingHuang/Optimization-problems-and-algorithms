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
    
    F = np.sum( np.abs(X), axis=1 ) + np.prod( np.abs(X), axis=1 )
    
    return F

#%% 參數設定
P = 100
D = 2
G = 500
w_max = 0.9
w_min = 0.2
c1 = 2
c2 = 2
ub = 50*np.ones([P, D])
lb = -50*np.ones([P,D])
k = 0.2

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
average_F = np.zeros(G)
watcher_local = np.zeros(G)
watcher_global = np.zeros([G, P, D])

#%% 迭代
for g in range(G):
    # 適應值計算
    F = fitness(X)
    average_F[g] = np.sum(F, axis=0) / P
    watcher_local[g] = X[0, 0]
    watcher_global[g] = X
    
    
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
    mask1 = X>ub
    mask2 = X<lb
    X[mask1] = ub[mask1]
    X[mask2] = lb[mask2]
    
#%% 作圖
plt.figure()
plt.title('gbest_F')
plt.plot(loss_curve)
plt.grid()
plt.semilogy(base=10)
plt.xlabel('Iteration')
plt.ylabel('Fitness')

plt.figure()
plt.title('Comparison')
plt.plot(loss_curve, color='red', label='algorithm1')
plt.plot(0.01*loss_curve, color='blue', label='algorithm2')
plt.grid()
plt.legend()
plt.semilogy(base=10)
plt.xlabel('Iteration')
plt.ylabel('Fitness')

plt.figure()
plt.title('average_F')
plt.plot(average_F)
plt.grid()
plt.semilogy(base=10)
plt.xlabel('Iteration')
plt.ylabel('Fitness')

plt.figure()
plt.title('X[0, 0]')
plt.plot(watcher_local)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Value')

plt.figure()
plt.title('Final_X')
plt.scatter(watcher_global[-1, :, 0], watcher_global[-1, :, 1])
[plt.text(watcher_global[-1, i, 0], watcher_global[-1, i, 1], 'X'+str(i+1)) for i in range(P)]
plt.grid()
plt.xlabel('X[:, 0]')
plt.ylabel('X[:, 1]')