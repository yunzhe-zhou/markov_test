from markov_test.REAL import *
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='mdn')
parser.add_argument('-L', '--L', type=int, default=3)
parser.add_argument('-B', '--B', type=int, default=1000)
parser.add_argument('-M', '--M', type=int, default=100)
parser.add_argument('-Q', '--Q', type=int, default=10)
parser.add_argument('-lr', '--lr', type=float, default=0.005)
parser.add_argument('-num_h', '--num_h', type=int, default=40)
parser.add_argument('-n_iter', '--n_iter', type=int, default=6000)
args0 = parser.parse_args()

string = "mdn_real1_run_cv" + "_L_" + str(args0.L)+ "_B_" + str(args0.B) + "_M_" + str(args0.M) + "_Q_" + str(args0.Q) + "_lr_" + str(args0.lr) + "_num_h_" + str(args0.num_h) + "_n_iter_" + str(args0.n_iter) 


df = pd.read_csv("data/7cityMonth.dat", sep = " ", header=None)
df = np.array(df)

trend_ls=[]
for i in range(12):
    trend=np.zeros(df.shape[1])
    for j in range(45):
        trend=trend+df[i+j*12,:]
    trend=trend/45
    trend_ls.append(trend)

series=df 
for i in range(12):
    for j in range(45):
        series[i+j*12,:]=series[i+j*12,:]-trend_ls[i]

class Setting:
    def __init__(self):
        self.T = series.shape[0]
        self.x_dims=series.shape[1]
        self.L=args0.L
        self.B=args0.B
        self.M=args0.M
        self.Q=args0.Q
        self.lr=args0.lr
        self.show=False
        self.h1=args0.num_h; self.k1=None; self.n1=args0.n_iter
        self.h2=args0.num_h; self.k2=None; self.n2=args0.n_iter
        self.h3=args0.num_h; self.k3=None; self.n3=args0.n_iter
        self.h4=args0.num_h; self.k4=None; self.n4=args0.n_iter
        self.h5=args0.num_h; self.k5=None; self.n5=args0.n_iter
        self.h6=args0.num_h; self.k6=None; self.n6=args0.n_iter
        self.cv_numk = [1,3,5,7]
config=Setting()

pvalue_ls=[]
for i in range(12):
    config.test_lag=int(i+1)
    while True:
        try:
            k_forward, k_backward = real_cv(config,series)
            config.k1 = k_forward; config.k2 = k_forward; config.k3 = k_forward
            config.k4 = k_backward; config.k5 = k_backward; config.k6 = k_backward; 
            pvalue=real(config,series)
            break
        except:
            pvalue = None
    pvalue_ls.append(pvalue)
    np.save("result/"+string,pvalue_ls)