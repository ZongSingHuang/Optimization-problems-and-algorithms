# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:09:03 2021

@author: zongsing.huang
"""
# =============================================================================
# 最佳解的適應值為0
# 統計分析->mean, std
# 即時更新pbest, gbest
# 提早終止:若gbest_F的改善率小於0.1%連續發生0.1*G次，就結束計算
# 用semilogy畫圖
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums

def fitness(X):
    # Schwefel2.22
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    F = np.sum( np.abs(X), axis=1 ) + np.prod( np.abs(X), axis=1 )
    
    return F

#%% 參數設定
P = 30
D = 30
G = 500
k = 0.2
w_max = 0.9
w_min = 0.2
c1 = 2
c2 = 2
ub = 10*np.ones([D])
lb = -10*np.ones([D])
T = 10
SW = True # 提早終止

#%% 初始化
v_max = k*(ub-lb)*np.ones([P, D])
v_min = -1*v_max
loss_curve = np.zeros([T, G])
statistical_experiment = np.zeros(T)

#%% 迭代
for t in range(T):
    X = np.random.uniform(low=lb, high=ub, size=[P, D])
    V = np.zeros([P, D])
    pbest_X = np.zeros([P, D])
    pbest_F = np.ones(P)*np.inf
    gbest_X = np.zeros(D)
    gbest_F = np.inf
    early_stopping = 0
    
    for g in range(G):
        for p in range(P):
            # 適應值計算
            F = fitness(X[p])
            
            # 更新pbest
            if F<pbest_F[p]:
                pbest_X[p] = X[p]
                pbest_F[p] = F
            
            # 更新gbest
            if F<gbest_F:
                gbest_X = X[p]
                gbest_F = F
                early_stopping = 0
            
            # 更新w
            w = w_max - g*(w_max-w_min)/G
            
            # 更新V
            r1 = np.random.uniform(size=[D])
            r2 = np.random.uniform(size=[D])
            V[p] = w*V[p] + c1*r1*(pbest_X[p]-X[p]) + c2*r2*(gbest_X-X[p])
            # 邊界處理
            mask1 = V[p]>v_max[p]
            mask2 = V[p]<v_min[p]
            V[p, mask1] = v_max[p, mask1]
            V[p, mask2] = v_min[p, mask2]
            
            # 更新X
            X[p] = X[p] + V[p]
            mask1 = X[p]>ub
            mask2 = X[p]<lb
            X[p, mask1] = ub[mask1]
            X[p, mask2] = lb[mask2]
            
        loss_curve[t, g] = gbest_F
        if np.abs(loss_curve[t, g]-loss_curve[t, g-1]) / np.abs(loss_curve[t, g])<=1e-3:
            early_stopping = early_stopping + 1
        if early_stopping>=0.1*G and SW==True:
            break
        
    statistical_experiment[t] = gbest_F
    
#%% 作圖
if SW==False:
    plt.figure()
    plt.plot(loss_curve.mean(axis=0))
    plt.grid()
    plt.semilogy(base=10)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
else:
    # 由於每次提早終止的次代都不同，導致loss_curve長度不一致
    # 因此採取的作畫方法為，取T次適應值最好的畫圖
    idx = np.argmin(statistical_experiment)
    plt.figure()
    plt.plot(loss_curve[idx])
    plt.grid()
    plt.semilogy(base=10)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')

#%% 統計分析
mean = statistical_experiment.mean()
std = statistical_experiment.std()

#%% Wilcoxon ranksum
assum1 = 0.01*statistical_experiment
assum2 = statistical_experiment
_, pvalue = ranksums(assum1, assum2)
if pvalue<0.05:
    print('assum1 better than assum2~')
