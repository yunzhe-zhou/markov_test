import os
import sys
import argparse

parser = argparse.ArgumentParser(description='mdn')
parser.add_argument('-lr', '--lr', type=float, default=0.005)
parser.add_argument('-num_h', '--num_h', type=int, default=40)
parser.add_argument('-n_iter', '--n_iter', type=int, default=6000)
parser.add_argument('-L', '--L', type=int, default=3)
parser.add_argument('-B', '--B', type=int, default=1000)
parser.add_argument('-M', '--M', type=int, default=100)
parser.add_argument('-Q', '--Q', type=int, default=10)
args0 = parser.parse_args()

string = "mdn_real3_run_cv" + "_L_" + str(args0.L)+ "_B_" + str(args0.B) + "_M_" + str(args0.M) + "_Q_" + str(args0.Q) + "_lr_" + str(args0.lr) + "_num_h_" + str(args0.num_h) + "_n_iter_" + str(args0.n_iter) 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers 
import tensorflow_probability as tfp
import random
import pandas as pd
import copy
from markov_test._core_test_fun import *
from markov_test._QRF import *
from markov_test._DGP_Ohio import *
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import time
import math
from datetime import datetime
import logging
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
import random
from scipy import stats
from collections import defaultdict
import warnings
from scipy.stats import rankdata
import xlwt
from tempfile import TemporaryFile
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

tfd = tfp.distributions
tf.keras.backend.set_floatx('float64')


def real_cv(config,data):
    seed = 4
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    # sim
    N = config.N; T = config.T; x_dims=config.x_dims; B=config.B; L=config.L; Q=config.Q
    test_lag=config.test_lag; M=config.M;sd_G = config.sd_G; first_T=config.first_T; lr=config.lr
    cv_numk = config.cv_numk
    
    testing_data =data

    series_ls=[np.concatenate((a[0],a[1]),1)   for a in testing_data]
    series_ls_S=[a[0]   for a in testing_data]
    Z_ls=[]
    X_ls=[]
    Z_ls_S=[]
    X_ls_S=[]
    for n in range(N):
        series=series_ls[n]
        series_S=series_ls_S[n]
        series_rep=np.concatenate((series,series),axis=0)
        series_rep_S=np.concatenate((series_S,series_S),axis=0)
        for k in range(T):
            if k==0:
                basis=series_rep[k:(k+test_lag),].reshape([1,-1])
                basis_S=series_rep_S[k:(k+test_lag),].reshape([1,-1])
            else:
                add=series_rep[k:(k+test_lag),].reshape([1,-1])
                basis=np.concatenate((basis,add),axis=0)
                add_S=series_rep_S[k:(k+test_lag),].reshape([1,-1])
                basis_S=np.concatenate((basis_S,add_S),axis=0)
        Z_basis=basis
        X_basis=series
        Z_ls.append(Z_basis)
        X_ls.append(X_basis)
        Z_basis_S=basis_S
        X_basis_S=series_S
        Z_ls_S.append(Z_basis_S)
        X_ls_S.append(X_basis_S)
    
    kf = KFold(n_splits=L)
    kf.get_n_splits(zeros(N)) 
    
    for train_index, test_index in kf.split(series_ls):
        forward_z_train=[Z_ls[i][0:int(T-test_lag),:] for i in train_index]
        forward_z_train=np.concatenate(forward_z_train,axis=0)
        forward_x_train=[X_ls_S[i][test_lag:T,:] for i in train_index]
        forward_x_train=np.concatenate(forward_x_train,axis=0)    

        backward_z_train=[Z_ls[i][1:int(T-test_lag+1),:]  for i in train_index]
        backward_z_train=np.concatenate(backward_z_train,axis=0)
        backward_x_train=[X_ls[i][0:int(T-test_lag),:] for i in train_index]
        backward_x_train=np.concatenate(backward_x_train,axis=0)   
        
    f_cv_len = forward_z_train.shape[0]//2
    forward_z_train_cv = forward_z_train[:f_cv_len,:]
    forward_z_test_cv = forward_z_train[f_cv_len:,:]
    forward_x_train_cv = forward_x_train[:f_cv_len,:]
    forward_x_test_cv = forward_x_train[f_cv_len:,:]

    b_cv_len = backward_z_train.shape[0]//2
    backward_z_train_cv = backward_z_train[:b_cv_len,:]
    backward_z_test_cv = backward_z_train[b_cv_len:,:]
    backward_x_train_cv = backward_x_train[:b_cv_len,:]
    backward_x_test_cv = backward_x_train[b_cv_len:,:]
    
    cv_error_ls1 = []
    cv_error_ls2 = []

    for numk in cv_numk:
        # forward dim 1
        note=0
        z_train = forward_z_train_cv
        x_train0 = forward_x_train_cv
        x_train=x_train0[:,[note]]

        hidden_units=config.h1; k_mixt=numk ; num_iter=config.n1
        sample=MDN_learning1(z_train,x_train,forward_z_test_cv,hidden_units,k_mixt,M,num_iter,lr)

        x_fake_ls1=tf.transpose(sample)
        x_fake_ls1=[tf.reshape(a,shape=[-1,1]) for a in x_fake_ls1]
        
        ###### dim 2 binary
        note=1
        z_train=forward_z_train_cv
        z_train=np.concatenate((z_train,forward_x_train_cv[:,:note]),1)
        x_train=forward_x_train_cv[:,[note]]
        x_train_hote= np.float64(x_train!=0)
        clf = LogisticRegression(random_state=0).fit(z_train, x_train_hote.reshape([-1,]))
        prob_ls1=[]
        for k in range(len(x_fake_ls1)):
            x_array=np.array(x_fake_ls1[k])
            z_array=np.tile(forward_z_test_cv[[k],:],[x_array.shape[0],1])
            predictor=np.concatenate((z_array,x_array),1)
            prob=clf.predict_proba(predictor)
            prob_ls1.append(prob)
        b_ls1=[]
        for k in range(len(x_fake_ls1)):
            b1=np.random.binomial(1,prob_ls1[k][:,[1]])
            b_ls1.append(b1)
            
        ######  dim 2 curve
        note=1
        z_train=forward_z_train_cv
        z_train=np.concatenate((z_train,forward_x_train_cv[:,:note]),1)
        x_train=forward_x_train_cv[:,[note]]
        z_train=z_train[x_train[:,0]!=0,:]
        x_train=x_train[x_train[:,0]!=0,:]
        x_train = x_train
        z_train = z_train

        hidden_units=config.h2; k_mixt=numk; num_iter=config.n2
        sample=MDN_learning2(z_train,x_train,forward_z_test_cv,hidden_units,k_mixt,M,num_iter,x_fake_ls1,b_ls1,lr)
        x_fake_ls2=tf.transpose(sample)
        x_fake_ls2=[tf.reshape(a,shape=[-1,1]) for a in x_fake_ls2]

        x_fake_ls12=[]
        for k in range(len(x_fake_ls1)):
            x_fake_ls12.append(tf.concat([x_fake_ls1[k],x_fake_ls2[k]],1))
            
            
        ###### dim 3 binary
        note=2
        z_train=forward_z_train_cv
        z_train=np.concatenate((z_train,forward_x_train_cv[:,:note]),1)
        x_train=forward_x_train_cv[:,[note]]
        x_train_hote= np.float64(x_train!=0)
        clf = LogisticRegression(random_state=0).fit(z_train, x_train_hote.reshape([-1,]))
        prob_ls2=[]
        for k in range(len(x_fake_ls12)):
            x_array=np.array(x_fake_ls12[k])
            z_array=np.tile(forward_z_test_cv[[k],:],[x_array.shape[0],1])
            predictor=np.concatenate((z_array,x_array),1)
            prob=clf.predict_proba(predictor)
            prob_ls2.append(prob)
        b_ls2=[]
        for k in range(len(x_fake_ls12)):
            b2=np.random.binomial(1,prob_ls2[k][:,[1]])
            b_ls2.append(b2)

        ######  dim 3 curve
        note=2
        z_train=forward_z_train_cv
        z_train=np.concatenate((z_train,forward_x_train_cv[:,:note]),1)
        x_train=forward_x_train_cv[:,[note]]
        z_train=z_train[x_train[:,0]!=0,:]
        x_train=x_train[x_train[:,0]!=0,:]

        hidden_units=config.h3; k_mixt=numk; num_iter=config.n3
        sample=MDN_learning2(z_train,x_train,forward_z_test_cv,hidden_units,k_mixt,M,num_iter,x_fake_ls12,b_ls2,lr)
        x_fake_ls3=tf.transpose(sample)
        x_fake_ls3=[tf.reshape(a,shape=[-1,1]) for a in x_fake_ls3]

        x_fake_ls123=[]
        for k in range(len(x_fake_ls1)):
            x_fake_ls123.append(tf.concat([x_fake_ls12[k],x_fake_ls3[k]],1))

        sim_forward=np.array(x_fake_ls123)
        
        # backward dim 1
        note=0
        z_train = backward_z_train_cv
        x_train0 = backward_x_train_cv
        x_train=x_train0[:,[note]]

        hidden_units=config.h4; k_mixt=numk; num_iter=config.n4
        sample=MDN_learning1(z_train,x_train,backward_z_test_cv,hidden_units,k_mixt,M,num_iter,lr)

        x_fake_back_ls1=tf.transpose(sample)
        x_fake_back_ls1=[tf.reshape(a,shape=[-1,1]) for a in x_fake_back_ls1]


        ###### dim 2 binary
        note=1
        z_train=backward_z_train_cv
        z_train=np.concatenate((z_train,backward_x_train_cv[:,:note]),1)
        x_train=backward_x_train_cv[:,[note]]
        x_train_hote= np.float64(x_train!=0)
        clf = LogisticRegression(random_state=0).fit(z_train, x_train_hote.reshape([-1,]))
        prob_ls1=[]
        for k in range(len(x_fake_back_ls1)):
            x_array=np.array(x_fake_back_ls1[k])
            z_array=np.tile(backward_z_test_cv[[k],:],[x_array.shape[0],1])
            predictor=np.concatenate((z_array,x_array),1)
            prob=clf.predict_proba(predictor)
            prob_ls1.append(prob)
        b_ls1=[]
        for k in range(len(x_fake_back_ls1)):
            b1=np.random.binomial(1,prob_ls1[k][:,[1]])
            b_ls1.append(b1)

        ######  dim 2 curve
        note=1
        z_train=backward_z_train_cv
        z_train=np.concatenate((z_train,backward_x_train_cv[:,:note]),1)
        x_train=backward_x_train_cv[:,[note]]
        z_train=z_train[x_train[:,0]!=0,:]
        x_train=x_train[x_train[:,0]!=0,:]

        hidden_units=config.h5; k_mixt=numk; num_iter=config.n5
        sample=MDN_learning2(z_train,x_train,backward_z_test_cv,hidden_units,k_mixt,M,num_iter,x_fake_back_ls1,b_ls1,lr)
        x_fake_back_ls2=tf.transpose(sample)
        x_fake_back_ls2=[tf.reshape(a,shape=[-1,1]) for a in x_fake_back_ls2]

        x_fake_back_ls12=[]
        for k in range(len(x_fake_back_ls1)):
            x_fake_back_ls12.append(tf.concat([x_fake_back_ls1[k],x_fake_back_ls2[k]],1))

        ###### dim 3 binary
        note=2
        z_train=backward_z_train_cv
        z_train=np.concatenate((z_train,backward_x_train_cv[:,:note]),1)
        x_train=backward_x_train_cv[:,[note]]
        x_train_hote= np.float64(x_train!=0)
        clf = LogisticRegression(random_state=0).fit(z_train, x_train_hote.reshape([-1,]))
        prob_ls2=[]
        for k in range(len(x_fake_back_ls12)):
            x_array=np.array(x_fake_back_ls12[k])
            z_array=np.tile(backward_z_test_cv[[k],:],[x_array.shape[0],1])
            predictor=np.concatenate((z_array,x_array),1)
            prob=clf.predict_proba(predictor)
            prob_ls2.append(prob)
        b_ls2=[]
        for k in range(len(x_fake_back_ls12)):
            b2=np.random.binomial(1,prob_ls2[k][:,[1]])
            b_ls2.append(b2)

        ######  dim 3 curve
        note=2
        z_train=backward_z_train_cv
        z_train=np.concatenate((z_train,backward_x_train_cv[:,:note]),1)
        x_train=backward_x_train_cv[:,[note]]
        z_train=z_train[x_train[:,0]!=0,:]
        x_train=x_train[x_train[:,0]!=0,:]

        hidden_units=config.h6; k_mixt=numk; num_iter=config.n6
        sample=MDN_learning2(z_train,x_train,backward_z_test_cv,hidden_units,k_mixt,M,num_iter,x_fake_back_ls12,b_ls2,lr)
        x_fake_back_ls3=tf.transpose(sample)
        x_fake_back_ls3=[tf.reshape(a,shape=[-1,1]) for a in x_fake_back_ls3]

        x_fake_back_ls123=[]
        for k in range(len(x_fake_back_ls1)):
            x_fake_back_ls123.append(tf.concat([x_fake_back_ls12[k],x_fake_back_ls3[k]],1))

        ###### dim 4 softmax
        note=3
        z_train=backward_z_train_cv
        z_train=np.concatenate((z_train,backward_x_train_cv[:,:note]),1)
        x_train=backward_x_train_cv[:,[note]]
        y=np.int64(x_train)
        clf = LogisticRegression(random_state=0).fit(z_train, y)
        prob_ls3=[]
        for k in range(len(x_fake_back_ls123)):
            x_array=np.array(x_fake_back_ls123[k])
            z_array=np.tile(backward_z_test_cv[[k],:],[x_array.shape[0],1])
            predictor=np.concatenate((z_array,x_array),1)
            prob=clf.predict_proba(predictor)
            prob_ls3.append(prob)
        b_ls3=[]
        for k in range(len(x_fake_back_ls123)):
            b3=[np.random.multinomial(1,prob_ls3[k][a]) for a in range(len(prob_ls3[k]))]
            b3=np.array(b3)
            b3=np.argmax(b3,1).reshape([-1,1])
            b_ls3.append(b3)

        x_fake_back_ls1234=[]
        for k in range(len(x_fake_back_ls1)):
            x_fake_back_ls1234.append(tf.concat([x_fake_back_ls123[k],b_ls3[k]],1))

        sim_backward=np.array(x_fake_back_ls1234)

        
        pred_forward = np.mean(sim_forward,1)
        cv_error1 = np.mean((pred_forward-forward_x_test_cv)**2)

        pred_backward = np.mean(sim_backward,1)
        cv_error2 = np.mean((pred_backward-backward_x_test_cv)**2)

        cv_error_ls1.append(cv_error1)
        cv_error_ls2.append(cv_error2)

    return cv_numk[np.argmin(cv_error_ls1)], cv_numk[np.argmin(cv_error_ls2)]

def real(config,data):
    seed = 4
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    # sim
    N = config.N; T = config.T; x_dims=config.x_dims; B=config.B; L=config.L; Q=config.Q
    test_lag=config.test_lag; M=config.M;sd_G = config.sd_G; first_T=config.first_T; lr=config.lr

    testing_data =data

    series_ls=[np.concatenate((a[0],a[1]),1)   for a in testing_data]
    series_ls_S=[a[0]   for a in testing_data]
    Z_ls=[]
    X_ls=[]
    Z_ls_S=[]
    X_ls_S=[]
    for n in range(N):
        series=series_ls[n]
        series_S=series_ls_S[n]
        series_rep=np.concatenate((series,series),axis=0)
        series_rep_S=np.concatenate((series_S,series_S),axis=0)
        for k in range(T):
            if k==0:
                basis=series_rep[k:(k+test_lag),].reshape([1,-1])
                basis_S=series_rep_S[k:(k+test_lag),].reshape([1,-1])
            else:
                add=series_rep[k:(k+test_lag),].reshape([1,-1])
                basis=np.concatenate((basis,add),axis=0)
                add_S=series_rep_S[k:(k+test_lag),].reshape([1,-1])
                basis_S=np.concatenate((basis_S,add_S),axis=0)
        Z_basis=basis
        X_basis=series
        Z_ls.append(Z_basis)
        X_ls.append(X_basis)
        Z_basis_S=basis_S
        X_basis_S=series_S
        Z_ls_S.append(Z_basis_S)
        X_ls_S.append(X_basis_S)

    uv = [randn(B, x_dims), randn(B, x_dims+1)]
    char_values, obs_ys = [np.zeros([N, T, B]) for i in range(4)], [
        np.zeros([N, T, B]) for i in range(4)]
    kf = KFold(n_splits=L)
    kf.get_n_splits(zeros(N)) 

    sim_forward_ls=[]
    sim_backward_ls=[]
    RF_forward_ls=[]
    RF_backward_ls=[]

    for train_index, test_index in kf.split(series_ls):
        forward_z_train=[Z_ls[i][0:int(T-test_lag),:] for i in train_index]
        forward_z_train=np.concatenate(forward_z_train,axis=0)
        forward_x_train=[X_ls_S[i][test_lag:T,:] for i in train_index]
        forward_x_train=np.concatenate(forward_x_train,axis=0)    

        backward_z_train=[Z_ls[i][1:int(T-test_lag+1),:]  for i in train_index]
        backward_z_train=np.concatenate(backward_z_train,axis=0)
        backward_x_train=[X_ls[i][0:int(T-test_lag),:] for i in train_index]
        backward_x_train=np.concatenate(backward_x_train,axis=0)   

        z_test0=[Z_ls[i] for i in test_index]
        z_test0=np.concatenate(z_test0,axis=0)

        # forward dim 1
        note=0
        z_train = forward_z_train
        x_train0 = forward_x_train
        x_train=x_train0[:,[note]]

        hidden_units=config.h1; k_mixt=config.k1 ; num_iter=config.n1
        sample=MDN_learning1(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,lr)

        x_fake_ls1=tf.transpose(sample)
        x_fake_ls1=[tf.reshape(a,shape=[-1,1]) for a in x_fake_ls1]

        ###### dim 2 binary
        note=1
        z_train=forward_z_train
        z_train=np.concatenate((z_train,forward_x_train[:,:note]),1)
        x_train=forward_x_train[:,[note]]
        x_train_hote= np.float64(x_train!=0)
        clf = LogisticRegression(random_state=0).fit(z_train, x_train_hote.reshape([-1,]))
        prob_ls1=[]
        for k in range(len(x_fake_ls1)):
            x_array=np.array(x_fake_ls1[k])
            z_array=np.tile(z_test0[[k],:],[x_array.shape[0],1])
            predictor=np.concatenate((z_array,x_array),1)
            prob=clf.predict_proba(predictor)
            prob_ls1.append(prob)
        b_ls1=[]
        for k in range(len(x_fake_ls1)):
            b1=np.random.binomial(1,prob_ls1[k][:,[1]])
            b_ls1.append(b1)

        ######  dim 2 curve
        note=1
        z_train=forward_z_train
        z_train=np.concatenate((z_train,forward_x_train[:,:note]),1)
        x_train=forward_x_train[:,[note]]
        z_train=z_train[x_train[:,0]!=0,:]
        x_train=x_train[x_train[:,0]!=0,:]
        x_train = x_train
        z_train = z_train

        hidden_units=config.h2; k_mixt=config.k2; num_iter=config.n2
        sample=MDN_learning2(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,x_fake_ls1,b_ls1,lr)
        x_fake_ls2=tf.transpose(sample)
        x_fake_ls2=[tf.reshape(a,shape=[-1,1]) for a in x_fake_ls2]

        x_fake_ls12=[]
        for k in range(len(x_fake_ls1)):
            x_fake_ls12.append(tf.concat([x_fake_ls1[k],x_fake_ls2[k]],1))

        ###### dim 3 binary
        note=2
        z_train=forward_z_train
        z_train=np.concatenate((z_train,forward_x_train[:,:note]),1)
        x_train=forward_x_train[:,[note]]
        x_train_hote= np.float64(x_train!=0)
        clf = LogisticRegression(random_state=0).fit(z_train, x_train_hote.reshape([-1,]))
        prob_ls2=[]
        for k in range(len(x_fake_ls12)):
            x_array=np.array(x_fake_ls12[k])
            z_array=np.tile(z_test0[[k],:],[x_array.shape[0],1])
            predictor=np.concatenate((z_array,x_array),1)
            prob=clf.predict_proba(predictor)
            prob_ls2.append(prob)
        b_ls2=[]
        for k in range(len(x_fake_ls12)):
            b2=np.random.binomial(1,prob_ls2[k][:,[1]])
            b_ls2.append(b2)

        ######  dim 3 curve
        note=2
        z_train=forward_z_train
        z_train=np.concatenate((z_train,forward_x_train[:,:note]),1)
        x_train=forward_x_train[:,[note]]
        z_train=z_train[x_train[:,0]!=0,:]
        x_train=x_train[x_train[:,0]!=0,:]

        hidden_units=config.h3; k_mixt=config.k3; num_iter=config.n3
        sample=MDN_learning2(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,x_fake_ls12,b_ls2,lr)
        x_fake_ls3=tf.transpose(sample)
        x_fake_ls3=[tf.reshape(a,shape=[-1,1]) for a in x_fake_ls3]

        x_fake_ls123=[]
        for k in range(len(x_fake_ls1)):
            x_fake_ls123.append(tf.concat([x_fake_ls12[k],x_fake_ls3[k]],1))

        sim_forward=np.array(x_fake_ls123)

        # backward dim 1
        note=0
        z_train = backward_z_train
        x_train0 = backward_x_train
        x_train=x_train0[:,[note]]

        hidden_units=config.h4; k_mixt=config.k4 ; num_iter=config.n4
        sample=MDN_learning1(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,lr)

        x_fake_back_ls1=tf.transpose(sample)
        x_fake_back_ls1=[tf.reshape(a,shape=[-1,1]) for a in x_fake_back_ls1]


        ###### dim 2 binary
        note=1
        z_train=backward_z_train
        z_train=np.concatenate((z_train,backward_x_train[:,:note]),1)
        x_train=backward_x_train[:,[note]]
        x_train_hote= np.float64(x_train!=0)
        clf = LogisticRegression(random_state=0).fit(z_train, x_train_hote.reshape([-1,]))
        prob_ls1=[]
        for k in range(len(x_fake_back_ls1)):
            x_array=np.array(x_fake_back_ls1[k])
            z_array=np.tile(z_test0[[k],:],[x_array.shape[0],1])
            predictor=np.concatenate((z_array,x_array),1)
            prob=clf.predict_proba(predictor)
            prob_ls1.append(prob)
        b_ls1=[]
        for k in range(len(x_fake_back_ls1)):
            b1=np.random.binomial(1,prob_ls1[k][:,[1]])
            b_ls1.append(b1)

        ######  dim 2 curve
        note=1
        z_train=backward_z_train
        z_train=np.concatenate((z_train,backward_x_train[:,:note]),1)
        x_train=backward_x_train[:,[note]]
        z_train=z_train[x_train[:,0]!=0,:]
        x_train=x_train[x_train[:,0]!=0,:]

        hidden_units=config.h5; k_mixt=config.k5; num_iter=config.n5
        sample=MDN_learning2(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,x_fake_back_ls1,b_ls1,lr)
        x_fake_back_ls2=tf.transpose(sample)
        x_fake_back_ls2=[tf.reshape(a,shape=[-1,1]) for a in x_fake_back_ls2]

        x_fake_back_ls12=[]
        for k in range(len(x_fake_back_ls1)):
            x_fake_back_ls12.append(tf.concat([x_fake_back_ls1[k],x_fake_back_ls2[k]],1))

        ###### dim 3 binary
        note=2
        z_train=backward_z_train
        z_train=np.concatenate((z_train,backward_x_train[:,:note]),1)
        x_train=backward_x_train[:,[note]]
        x_train_hote= np.float64(x_train!=0)
        clf = LogisticRegression(random_state=0).fit(z_train, x_train_hote.reshape([-1,]))
        prob_ls2=[]
        for k in range(len(x_fake_back_ls12)):
            x_array=np.array(x_fake_back_ls12[k])
            z_array=np.tile(z_test0[[k],:],[x_array.shape[0],1])
            predictor=np.concatenate((z_array,x_array),1)
            prob=clf.predict_proba(predictor)
            prob_ls2.append(prob)
        b_ls2=[]
        for k in range(len(x_fake_back_ls12)):
            b2=np.random.binomial(1,prob_ls2[k][:,[1]])
            b_ls2.append(b2)

        ######  dim 3 curve
        note=2
        z_train=backward_z_train
        z_train=np.concatenate((z_train,backward_x_train[:,:note]),1)
        x_train=backward_x_train[:,[note]]
        z_train=z_train[x_train[:,0]!=0,:]
        x_train=x_train[x_train[:,0]!=0,:]

        hidden_units=config.h6; k_mixt=config.k6; num_iter=config.n6
        sample=MDN_learning2(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,x_fake_back_ls12,b_ls2,lr)
        x_fake_back_ls3=tf.transpose(sample)
        x_fake_back_ls3=[tf.reshape(a,shape=[-1,1]) for a in x_fake_back_ls3]

        x_fake_back_ls123=[]
        for k in range(len(x_fake_back_ls1)):
            x_fake_back_ls123.append(tf.concat([x_fake_back_ls12[k],x_fake_back_ls3[k]],1))

        ###### dim 4 softmax
        note=3
        z_train=backward_z_train
        z_train=np.concatenate((z_train,backward_x_train[:,:note]),1)
        x_train=backward_x_train[:,[note]]
        y=np.int64(x_train)
        clf = LogisticRegression(random_state=0).fit(z_train, y)
        prob_ls3=[]
        for k in range(len(x_fake_back_ls123)):
            x_array=np.array(x_fake_back_ls123[k])
            z_array=np.tile(z_test0[[k],:],[x_array.shape[0],1])
            predictor=np.concatenate((z_array,x_array),1)
            prob=clf.predict_proba(predictor)
            prob_ls3.append(prob)
        b_ls3=[]
        for k in range(len(x_fake_back_ls123)):
            b3=[np.random.multinomial(1,prob_ls3[k][a]) for a in range(len(prob_ls3[k]))]
            b3=np.array(b3)
            b3=np.argmax(b3,1).reshape([-1,1])
            b_ls3.append(b3)

        x_fake_back_ls1234=[]
        for k in range(len(x_fake_back_ls1)):
            x_fake_back_ls1234.append(tf.concat([x_fake_back_ls123[k],b_ls3[k]],1))

        sim_backward=np.array(x_fake_back_ls1234)

        sim_forward_ls.append(sim_forward)
        sim_backward_ls.append(sim_backward)

    np.random.seed(seed)
    uv = [randn(B, x_dims), randn(B, x_dims+1)]
    char_values, obs_ys = [np.zeros([N, T, B]) for i in range(4)], [
        np.zeros([N, T, B]) for i in range(4)]
    zz=0
    for train_index, test_index in kf.split(series_ls):
        sim_forward=copy.deepcopy(sim_forward_ls[zz])
        sim_backward=copy.deepcopy(sim_backward_ls[zz])

        r_ls=[]

        base=np.matmul(sim_forward,uv[0].T)
        r1=np.mean(np.cos(base),1)
        r2=np.mean(np.sin(base),1)
        r_ls.append([r1,r2])

        base=np.matmul(sim_backward,uv[1].T)
        r1=np.mean(np.cos(base),1)
        r2=np.mean(np.sin(base),1)
        r_ls.append([r1,r2])

        zz=zz+1
        for i in range(2):  # forward / backward
            r = r_ls[i]
            char_values[0 + i][test_index] = r[0].reshape((len(test_index), T, B))
            char_values[2 + i][test_index] = r[1].reshape((len(test_index), T, B)) 

    estimated_cond_char = char_values
    series_array_ls=[]
    series_array_ls.append(np.array(series_ls_S))
    series_array_ls.append(np.array(series_ls))
    observed_cond_char = []
    for i in range(2):
        temp = series_array_ls[i].dot(uv[i].T)
        observed_cond_char += [cos(temp), sin(temp)]
    # combine the above two parts to get cond. corr. estimation.
    phi_R, psi_R, phi_I, psi_I = estimated_cond_char
    c_X, s_X, c_XA, s_XA = observed_cond_char

    # forward, t is the residual at time t
    left_cos_R = c_X - roll(phi_R, test_lag, 1)
    left_sin_I = s_X - roll(phi_I, test_lag, 1)
    # backward, t is the residual at time t
    right_cos_R = c_XA - roll(psi_R, -1, 1)
    right_sin_I = s_XA - roll(psi_I, -1, 1)        

    lam = []

    for q in range(2, Q + 1):
        shift = q + test_lag - 1
        startT = q + test_lag - 1
        lam_RR = multiply(
            left_cos_R, roll(
                right_cos_R, shift, 1))[
            :, startT:, :]
        lam_II = multiply(
            left_sin_I, roll(
                right_sin_I, shift, 1))[
            :, startT:, :]
        lam_IR = multiply(
            left_sin_I, roll(
                right_cos_R, shift, 1))[
            :, startT:, :]
        lam_RI = multiply(
            left_cos_R, roll(
                right_sin_I, shift, 1))[
            :, startT:, :]
        lam.append([lam_RR, lam_II, lam_IR, lam_RI])

    Sigma_q_s = Sigma_q(lam)  # a list (len = Q-1) 2B * 2B.
    S = S_hat(lam = lam, dims = [N, T], J = test_lag)  # Construct the test statistics
    pValues = bootstrap_p_value(Sigma_q_s, rep_times = int(1e3), test_stat = S) 
    return pValues

def MDN_learning1(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,lr,seed_utils=2021):
    if seed_utils < 2018:
        return np.nan
    np.random.seed(seed_utils)
    random.seed(seed_utils)
    tf.random.set_seed(seed_utils)

    hidden_dense = Dense(hidden_units, activation=tf.nn.relu,
                  kernel_regularizer=regularizers.l2(0.01))
    
    hidden = hidden_dense(z_train)
    alpha_dense = Dense(k_mixt, activation=tf.nn.softmax)
    alpha = alpha_dense(hidden)
    mu_dense = Dense(k_mixt, activation=None)
    mu = mu_dense((hidden))
    sigma_dense = Dense(k_mixt, activation=tf.nn.softplus,name='sigma')
    sigma=sigma_dense(hidden)
    gm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=alpha),components_distribution=tfd.Normal(loc=mu, scale=sigma))

    tvars = hidden_dense.trainable_variables + alpha_dense.trainable_variables +mu_dense.trainable_variables+sigma_dense.trainable_variables
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def step():
        with tf.GradientTape() as tape:
            #tape.watch(tvars)
            hidden = hidden_dense(z_train)
            alpha = alpha_dense(hidden)
            mu = mu_dense((hidden))
            sigma=sigma_dense(hidden)
            gm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=alpha),components_distribution=tfd.Normal(loc=mu, scale=sigma))
            loss = -tf.reduce_sum(gm.log_prob(tf.reshape(x_train,(-1,))))
            grads = tape.gradient(loss,tvars)
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        optimizer.apply_gradients(zip(grads, tvars)) 
        return loss

    for i in range(num_iter):
        loss=step()
        if np.isnan(loss):
            print("switch",seed_utils)
            sample=MDN_learning1(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,lr,seed_utils-1)
            return sample
        if i%100 == 0:
            print(str(i)+": "+str(np.double(loss)))

    hidden = hidden_dense(z_test0)
    alpha = alpha_dense(hidden)
    mu = mu_dense((hidden))
    sigma=sigma_dense(hidden)
    gm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=alpha),components_distribution=tfd.Normal(loc=mu, scale=sigma))
    sample=gm.sample(M)
    return sample

def MDN_learning2(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,x_sim,b_ls,lr,seed_utils=2021):
    if seed_utils < 2018:
        return np.nan
    np.random.seed(seed_utils)
    random.seed(seed_utils)
    tf.random.set_seed(seed_utils)

    hidden_dense = Dense(hidden_units, activation=tf.nn.relu,
                  kernel_regularizer=regularizers.l2(0.01))

    hidden = hidden_dense(z_train)
    alpha_dense = Dense(k_mixt, activation=tf.nn.softmax)
    alpha = alpha_dense(hidden)
    mu_dense = Dense(k_mixt, activation=None)
    mu = mu_dense((hidden))
    sigma_dense = Dense(k_mixt, activation=tf.nn.softplus,name='sigma')
    sigma=sigma_dense(hidden)
    gm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=alpha),components_distribution=tfd.Normal(loc=mu, scale=sigma))

    tvars = hidden_dense.trainable_variables + alpha_dense.trainable_variables +mu_dense.trainable_variables+sigma_dense.trainable_variables
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def step():
        with tf.GradientTape() as tape:
            #tape.watch(tvars)
            hidden = hidden_dense(z_train)
            alpha = alpha_dense(hidden)
            mu = mu_dense((hidden))
            sigma=sigma_dense(hidden)
            gm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=alpha),components_distribution=tfd.Normal(loc=mu, scale=sigma))
            loss = -tf.reduce_sum(gm.log_prob(tf.reshape(x_train,(-1,))))
            grads = tape.gradient(loss,tvars)
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        optimizer.apply_gradients(zip(grads, tvars)) 
        return loss

    for i in range(num_iter):
        loss=step()
        if np.isnan(loss):
            print("switch",seed_utils)
            sample=MDN_learning2(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,x_sim,lr,b_ls,seed_utils-1)
            return sample
        if i%100 == 0:
            print(str(i)+": "+str(np.double(loss)))


    z_test_ls=[]
    for k in range(z_test0.shape[0]):
        tiled_z = tf.tile(z_test0[[k],:], [M, 1])
        z_test_ls.append(tf.concat([tiled_z,x_sim[k]], axis=1))
    z_test=tf.concat(z_test_ls,axis=0)
    hidden = hidden_dense(z_test)
    alpha = alpha_dense(hidden)
    mu = mu_dense((hidden))
    sigma=sigma_dense(hidden)
    gm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=alpha),components_distribution=tfd.Normal(loc=mu, scale=sigma))

    tf.random.set_seed(2021)
    sample=tf.transpose(gm.sample(1))
    b_ls_add=np.concatenate(b_ls,axis=0)
    sample=sample*b_ls_add
    sample=tf.reshape(sample,[len(z_test_ls),-1])
    sample=tf.transpose(sample)
    return sample

class Setting:
    def __init__(self):
        self.N = 6
        self.T = 1100
        self.x_dims=3
        self.test_lag=None
        self.L=args0.L
        self.B=1000
        self.M=100
        self.sd_G=3
        self.lr=args0.lr
        self.first_T=10
        self.Q=10
        self.h1=args0.num_h; self.k1=None; self.n1=args0.n_iter
        self.h2=args0.num_h; self.k2=None; self.n2=args0.n_iter
        self.h3=args0.num_h; self.k3=None; self.n3=args0.n_iter
        self.h4=args0.num_h; self.k4=None; self.n4=args0.n_iter
        self.h5=args0.num_h; self.k5=None; self.n5=args0.n_iter
        self.h6=args0.num_h; self.k6=None; self.n6=args0.n_iter
        self.cv_numk = [1,3,5,7]
config=Setting()

data_raw=pd.read_csv("data/Data_Ohio.csv")
data0=np.array(data_raw)
data0=data0[:,1:]
T = config.T
data_J = []
for i in range(6):
    temp = data0[T * i : T * (i + 1)]
    temp = [temp[:, :3], temp[:, 3].reshape(-1, 1)]
    data_J.append(temp)
data= normalize(data_J)

pvalue_ls=[]
for i in range(12):
    config.test_lag=int(i+1)
    k_forward, k_backward = real_cv(config,data)
    config.k1 = k_forward; config.k2 = k_forward; config.k3 = k_forward
    config.k4 = k_backward; config.k5 = k_backward; config.k6 = k_backward; 
    pvalue=real(config,data)
    pvalue_ls.append(pvalue)
    np.save("result/"+string,pvalue_ls)