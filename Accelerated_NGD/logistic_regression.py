import matplotlib
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import numpy.linalg as la
from sklearn.datasets import load_svmlight_file

from optimizers import *
from loss_functions import logistic_loss, logistic_gradient

import time

sns.set(style="whitegrid", font_scale=1.2, context="talk", palette=sns.color_palette("bright"), color_codes=False)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
datasets = [
    "a9a"
    ,
    "cod-rna",
    "ijcnn1",
    "mushrooms",
    "phishing",
    "skin_nonskin",
    "w8a"
]

for dataset in datasets:
    data_path = './images/datasets/' + dataset
    if dataset == 'covtype':
        data_path += '.bz2'
    
    if dataset == 'ijcnn1' or dataset == 'rcv1' or dataset == 'real-sim' or dataset == 'imdb_sentiment':
        data_path += '.bz2'
    
    if dataset == 'covtype':
        it_max = 1000
        
    elif dataset == 'mushrooms':
        it_max = 3000
    elif dataset == 'w8a':
        it_max = 2000
    elif dataset == 'a9a':
        it_max = 1000
    elif dataset == 'phishing':
        it_max = 2000
    elif dataset == 'cod-rna':
        it_max = 1000
    else:
        it_max = 3000
    
    def logistic_smoothness(X):
        return 0.25 * np.max(la.eigvalsh(X.T @ X / X.shape[0]))
    
    data = load_svmlight_file(data_path)
    X, y = data[0].toarray(), data[1]
    if (np.unique(y) == [1, 2]).all():
        # Loss functions support only labels from {0, 1}
        y -= 1
    if (np.unique(y) == [-1, 1]).all():
        y = (y+1) / 2
    n, d = X.shape
    L = logistic_smoothness(X)
    l2 = L / n if dataset == 'covtype' else L / (10 * n) 
    w0 = np.zeros(d)
    
    def loss_func(w):
        return logistic_loss(w, X, y, l2)
    
    def grad_func(w):
        return logistic_gradient(w, X, y, l2)
    
    print(X.shape)
    
    print("Running Gd...")
    start_time = time.time()
    gd = Gd(lr=1 / L, loss_func=loss_func, grad_func=grad_func, it_max=it_max)
    gd.run(w0=w0)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    print("Running Adgd...")
    start_time = time.time()
    adgd = Adgd(loss_func=loss_func, grad_func=grad_func, eps=0, it_max=it_max)
    adgd.run(w0=w0)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    print("Running AdgdAccel...")
    start_time = time.time()
    ad_acc = AdgdAccel(loss_func=loss_func, grad_func=grad_func, it_max=it_max)
    ad_acc.run(w0=w0)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    print("Running NGD...")
    start_time = time.time()
    ngd = NGD(loss_func=loss_func, grad_func=grad_func, lr0=1e-3, eta0=0.2, eta1=0.15, beta=4.5, alpha=2, it_max=it_max)
    ngd.run(w0=w0)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    print("Running NGDn...")
    start_time = time.time()
    ngdn = NGDn(loss_func=loss_func, grad_func=grad_func, lr0=0.01, eta0=0.2, eta1=0.19, beta=0, alpha=3.0, gamma = 0.9, it_max=it_max)
    ngdn.run(w0=w0)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    print("Running NGDh...")
    start_time = time.time()
    ngdh = NGDh(loss_func=loss_func, grad_func=grad_func, lr0=0.01, eta0=0.2, eta1=0.19, beta=0, alpha=3.0, gamma = 0.9, it_max=it_max)
    ngdh.run(w0=w0)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    print("Running Nesterov...")
    start_time = time.time()
    nesterov = Nesterov(lr=1 / L, strongly_convex=True, mu=l2, loss_func=loss_func, grad_func=grad_func, it_max=it_max)
    nesterov.run(w0=w0)
    print("--- Nesterov Time: %s seconds ---" % (time.time() - start_time))

    print("Running Heavy Ball...")
    start_time = time.time()
    heavy_ball = HeavyBall(
        L=L, 
        mu=l2, 
        loss_func=loss_func, 
        grad_func=grad_func, 
        it_max=it_max
    )
    
    heavy_ball.run(w0=w0.copy())
    
    print("--- Heavy Ball Time: %s seconds ---" % (time.time() - start_time))
    
    
    optimizers = [gd, adgd, ad_acc, ngd, ngdn, ngdh, nesterov, heavy_ball]
    markers = [',', 'o', '^', '*', 'D', 'x', 's', 'v']
    labels = ['GD', 'AdGD', 'AdGD-accel', 'NGD','NGDn', 'NGDh', 'Nesterov-accel', "Heavy-ball-accel"]
    
    for opt, marker in zip(optimizers, markers):
        opt.compute_loss_on_iterates()
    f_star = np.min([np.min(opt.losses) for opt in optimizers])
    print(f_star)

    
    plt.figure(figsize=(8, 6))
    
    for opt, marker, label in zip(optimizers, markers, labels):
        opt.plot_losses(marker=marker, f_star=f_star, label=label)
        
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'$f(x_k) - f_*$')
    
    # Logic trích xuất và lưu riêng label ra 1 file (chỉ chạy đúng 1 lần cho dataset đầu tiên)
    if dataset == datasets[0]:
        handles, extracted_labels = plt.gca().get_legend_handles_labels()
        
        # Tạo một figure trống chỉ để chứa legend
        fig_legend = plt.figure(figsize=(10, 1.5))
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')
        
        # Sắp xếp legend theo chiều ngang để dễ nhìn
        ax_legend.legend(handles, extracted_labels, loc='center', ncol=4, frameon=True)
        fig_legend.savefig("./results/plot_legend_only.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig_legend)

    # Không gọi plt.legend() ở đây nữa để các plot chính không bị rối
    plt.tight_layout()
    plt.savefig(f"./results/{dataset}.pdf", format="pdf", bbox_inches="tight")
    plt.close()
