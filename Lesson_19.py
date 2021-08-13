# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:09:03 2021

@author: zongsing.huang
"""
# =============================================================================
# 在每一粒子附近作小幅度的擾動，每擾動一次就計算一次適應值，這樣就能獲得該粒子原始位置附近的平均適應值
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

def fitness(X):
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    i = 1e-3 # 適應函數的係數，不用理會
    
    F = -((1/((2*np.pi)**0.5))*np.exp(-0.5*((((X[:, 0]-1.5)*(X[:, 0]-1.5)+(X[:, 1]-1.5)*(X[:, 1]-1.5))/0.5)))
    +(2/((2*np.pi)**0.5))*np.exp(-0.5*((((X[:, 0]-0.5)*(X[:, 0]-0.5)+(X[:, 1]-0.5)*(X[:, 1]-0.5))/i))))
    
    return F

def fitness_VM(X):
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    i = 1e-3 # 適應函數的係數，不用理會
    H = 50 # 擾動次數
    delta = 0.05 # 擾動範圍
    threshold = 0.01 # 閥值
    P = X.shape[0]
    D = X.shape[1]
    
    # step1. 計算X的適應值
    original_F = -((1/((2*np.pi)**0.5))*np.exp(-0.5*((((X[:, 0]-1.5)*(X[:, 0]-1.5)+(X[:, 1]-1.5)*(X[:, 1]-1.5))/0.5)))
    +(2/((2*np.pi)**0.5))*np.exp(-0.5*((((X[:, 0]-0.5)*(X[:, 0]-0.5)+(X[:, 1]-0.5)*(X[:, 1]-0.5))/i))))
    
    # step2. 建立一個尺寸為[P, H]的矩陣，用來儲存擾動後X的適應值
    perturbed_F = np.zeros([P, H])
    
    # step3. 開始擾動，持續H次
    for h in range(H):
        # step3-1. 產生擾動後X
        perturbation_factor = np.random.uniform(low=-delta, high=delta, size=[P, D]) # 擾動因子
        perturbed_X = X + perturbation_factor # 擾動後X = X + 擾動因子
         
        # step3-2. 計算擾動後X的適應值
        perturbed_F[:, h] = -((1/((2*np.pi)**0.5))*np.exp(-0.5*((((perturbed_X[:, 0]-1.5)*(perturbed_X[:, 0]-1.5)+(perturbed_X[:, 1]-1.5)*(perturbed_X[:, 1]-1.5))/0.5)))
                 +(2/((2*np.pi)**0.5))*np.exp(-0.5*((((perturbed_X[:, 0]-0.5)*(perturbed_X[:, 0]-0.5)+(perturbed_X[:, 1]-0.5)*(perturbed_X[:, 1]-0.5))/i))))
        
    # step4. 計算最終的適應值
    F = ( np.mean(perturbed_F, axis=1) - original_F ) / original_F # |擾動後X的適應值之平均-X的適應值 / X的適應值|
    F = np.abs(F)
    
    # step5. 懲罰值
    mask = F>=threshold
    F[mask] = 200
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
ub = 10*np.ones([P, D])
lb = -10*np.ones([P, D])

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
    F = fitness_VM(X)
    
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