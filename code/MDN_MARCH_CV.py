import numpy as np
import random
import matplotlib.pyplot as plt
from _core_test_fun import *
from _QRF import *
from _DGP_Ohio import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers 
import tensorflow_probability as tfp
tfd = tfp.distributions

np.random.seed(2021)
random.seed(2021)
tf.random.set_seed(2021)

tf.keras.backend.set_floatx('float64')

def check_stationarity(config):
    lag=config.A1.shape[0]
    cov = 2*config.nstd*np.eye(config.x_dims)
    mu = np.zeros(config.x_dims)
    I=np.concatenate([np.eye(config.x_dims*(lag-1)),np.zeros([config.x_dims*(lag-1),config.x_dims])],axis=1)
    A=np.concatenate([np.concatenate([config.A1,config.A2,config.A3],axis=1),I],axis=0)
    eigen=np.max([np.sqrt(a.real**2+a.imag**2) for a in np.linalg.eig(A)[0]])
    print(eigen)

def generate_time_series(T=100, x_dims=3, nstd=1, test_lag=2, seed=2020,M=500,MC=True,show=True,first_T=1000,A1=0,A2=0,A3=0,B1=0,B2=0,B3=0,sim_type="var",constant=0.1):
    np.random.seed(seed)
    T0=T+first_T
    lag=A1.shape[0]
    cov = 2*nstd*np.eye(x_dims)
    mu = np.zeros(x_dims)

    series=np.random.multivariate_normal(mu, cov, lag)
    X_MC1=[]

    if sim_type=="var":
        for k in range(T0):
            if k>=lag:
                add_basis=np.matmul(A1,series[[k-1],:].T)+np.matmul(A2,series[[k-2],:].T)+np.matmul(A3,series[[k-3],:].T)
                add_basis_noise=add_basis.T+np.random.multivariate_normal(mu, cov, 1)
                series=np.concatenate((series,add_basis_noise),axis=0)
                if (MC==True) and (k>=first_T):
                    add_MC1=add_basis.T+np.random.multivariate_normal(mu, cov, M)
                    X_MC1.append(add_MC1)  
    elif sim_type=="march":
        for k in range(T0):
            if k>=lag:
                add_basis=constant+np.matmul(A1,(series[[k-1],:]**2).T)+np.matmul(A2,(series[[k-2],:]**2).T)+np.matmul(A3,(series[[k-3],:]**2).T)
                cov=2*nstd*np.eye(x_dims)*add_basis.T
                add_basis_noise=np.random.multivariate_normal(mu, cov, 1)
                series=np.concatenate((series,add_basis_noise),axis=0)
                if (MC==True) and (k>=first_T):
                    add_MC1=np.random.multivariate_normal(mu, cov, M)
                    X_MC1.append(add_MC1)  
    elif sim_type=="shred":
        for k in range(T0):
            if k>=lag:
                if np.sum(series[[k-1],:])<=0:
                    add_basis=np.matmul(A1,series[[k-1],:].T)+np.matmul(A2,series[[k-2],:].T)+np.matmul(A3,series[[k-3],:].T)
                else:
                    add_basis=np.matmul(B1,series[[k-1],:].T)+np.matmul(B2,series[[k-2],:].T)+np.matmul(B3,series[[k-3],:].T)
                add_basis_noise=add_basis.T+np.random.multivariate_normal(mu, cov, 1)
                series=np.concatenate((series,add_basis_noise),axis=0)
                if (MC==True) and (k>=first_T):
                    add_MC1=add_basis.T+np.random.multivariate_normal(mu, cov, M)
                    X_MC1.append(add_MC1)  

    series=series[first_T:T0,:]
    if sim_type=="march":
        series= np.matmul(series,B1.T)
    series_rep=np.concatenate((series,series),axis=0)
    for k in range(T):
        if k==0:
            basis=series_rep[k:(k+test_lag),].reshape([1,-1])
        else:
            add=series_rep[k:(k+test_lag),].reshape([1,-1])
            basis=np.concatenate((basis,add),axis=0)

    Z_ls=basis
    X_ls=series
    if sim_type=="march":
        X_MC1_ls=[np.matmul(a,B1.T) for a in X_MC1]
    else:
        X_MC1_ls=X_MC1
    series_ls=series

    if show==True:
        for k in range(x_dims):
            plt.plot(range(T),series[:,k])
            plt.show()

    if MC==True:
        return series_ls, X_ls, Z_ls, X_MC1_ls
    else:
        return series_ls, X_ls, Z_ls,0

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

def MDN_learning2(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,x_sim,lr,seed_utils=2021):
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
            sample=MDN_learning2(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,x_sim,lr,seed_utils-1)
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
    sample=tf.reshape(sample,[len(z_test_ls),-1])
    sample=tf.transpose(sample)
    return sample

def simulate_cv(seed,config):
    # sim
    T = config.T; x_dims=config.x_dims; nstd=config.nstd; 
    test_lag=config.test_lag;  L=config.L; B=config.B; 
    M=config.M; first_T=config.first_T; Q=config.Q; show=config.show
    K=int(T/L);N = config.L-1; lr=config.lr
    A1=config.A1;A2=config.A2;A3=config.A3
    B1=config.B1;B2=config.B2;B3=config.B3
    sim_type=config.sim_type; constant=config.constant
    cv_numk = config.cv_numk

    series_ls, X_ls, Z_ls, X_MC1_ls=generate_time_series(T,x_dims,nstd,test_lag,seed,M,False,show,first_T,A1,A2,A3,B1,B2,B3,sim_type,constant)
    
    np.random.seed(seed)
    
    n = 0

    forward_z_train= Z_ls[:int((n+1)*K),:]
    forward_z_train= forward_z_train[0:int(K*(n+1)-test_lag),:]

    forward_x_train= X_ls[:int((n+1)*K),:]
    forward_x_train= forward_x_train[test_lag:int(K*(n+1)),:]

    backward_z_train= Z_ls[:int((n+1)*K),:]
    backward_z_train= backward_z_train[1:int(K*(n+1)-test_lag+1),:]

    backward_x_train= X_ls[:int((n+1)*K),:]
    backward_x_train= backward_x_train[0:int(K*(n+1)-test_lag),:]

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

        ######  dim 2 curve
        note=1
        z_train=forward_z_train_cv
        z_train=np.concatenate((z_train,forward_x_train_cv[:,:note]),1)
        x_train=forward_x_train_cv[:,[note]]

        hidden_units=config.h2; k_mixt=numk; num_iter=config.n2
        sample=MDN_learning2(z_train,x_train,forward_z_test_cv,hidden_units,k_mixt,M,num_iter,x_fake_ls1,lr)
        x_fake_ls2=tf.transpose(sample)
        x_fake_ls2=[tf.reshape(a,shape=[-1,1]) for a in x_fake_ls2]

        x_fake_ls12=[]
        for k in range(len(x_fake_ls1)):
            x_fake_ls12.append(tf.concat([x_fake_ls1[k],x_fake_ls2[k]],1))

        ######  dim 3 curve
        note=2
        z_train=forward_z_train_cv
        z_train=np.concatenate((z_train,forward_x_train_cv[:,:note]),1)
        x_train=forward_x_train_cv[:,[note]]

        hidden_units=config.h3; k_mixt=numk; num_iter=config.n3
        sample=MDN_learning2(z_train,x_train,forward_z_test_cv,hidden_units,k_mixt,M,num_iter,x_fake_ls12,lr)
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

        hidden_units=config.h4; k_mixt=numk ; num_iter=config.n4
        sample=MDN_learning1(z_train,x_train,backward_z_test_cv,hidden_units,k_mixt,M,num_iter,lr)

        x_fake_back_ls1=tf.transpose(sample)
        x_fake_back_ls1=[tf.reshape(a,shape=[-1,1]) for a in x_fake_back_ls1]

        ######  dim 2 curve
        note=1
        z_train=backward_z_train_cv
        z_train=np.concatenate((z_train,backward_x_train_cv[:,:note]),1)
        x_train=backward_x_train_cv[:,[note]]

        hidden_units=config.h5; k_mixt=numk; num_iter=config.n5
        sample=MDN_learning2(z_train,x_train,backward_z_test_cv,hidden_units,k_mixt,M,num_iter,x_fake_back_ls1,lr)
        x_fake_back_ls2=tf.transpose(sample)
        x_fake_back_ls2=[tf.reshape(a,shape=[-1,1]) for a in x_fake_back_ls2]

        x_fake_back_ls12=[]
        for k in range(len(x_fake_back_ls1)):
            x_fake_back_ls12.append(tf.concat([x_fake_back_ls1[k],x_fake_back_ls2[k]],1))


        ######  dim 3 curve
        note=2
        z_train=backward_z_train_cv
        z_train=np.concatenate((z_train,backward_x_train_cv[:,:note]),1)
        x_train=backward_x_train_cv[:,[note]]

        hidden_units=config.h6; k_mixt=numk; num_iter=config.n6
        sample=MDN_learning2(z_train,x_train,backward_z_test_cv,hidden_units,k_mixt,M,num_iter,x_fake_back_ls12,lr)
        x_fake_back_ls3=tf.transpose(sample)
        x_fake_back_ls3=[tf.reshape(a,shape=[-1,1]) for a in x_fake_back_ls3]

        x_fake_back_ls123=[]
        for k in range(len(x_fake_back_ls1)):
            x_fake_back_ls123.append(tf.concat([x_fake_back_ls12[k],x_fake_back_ls3[k]],1))
        sim_backward=np.array(x_fake_back_ls123)

        pred_forward = np.mean(sim_forward,1)
        cv_error1 = np.mean((pred_forward-forward_x_test_cv)**2)

        pred_backward = np.mean(sim_backward,1)
        cv_error2 = np.mean((pred_backward-backward_x_test_cv)**2)

        cv_error_ls1.append(cv_error1)
        cv_error_ls2.append(cv_error2)

    return cv_numk[np.argmin(cv_error_ls1)], cv_numk[np.argmin(cv_error_ls2)]

def simulate(seed,config):
    # sim
    T = config.T; x_dims=config.x_dims; nstd=config.nstd; 
    test_lag=config.test_lag;  L=config.L; B=config.B; 
    M=config.M; first_T=config.first_T; Q=config.Q; show=config.show
    K=int(T/L);N = config.L-1; lr=config.lr
    A1=config.A1;A2=config.A2;A3=config.A3
    B1=config.B1;B2=config.B2;B3=config.B3
    sim_type=config.sim_type; constant=config.constant

    series_ls, X_ls, Z_ls, X_MC1_ls=generate_time_series(T,x_dims,nstd,test_lag,seed,M,False,show,first_T,A1,A2,A3,B1,B2,B3,sim_type,constant)
    
    np.random.seed(seed)
    
    uv = [randn(B, x_dims), randn(B, x_dims)]
    char_values, obs_ys = [np.zeros([N, K, B]) for i in range(4)], [
        np.zeros([N, K, B]) for i in range(4)]

    for n in range(N):
        forward_z_train= Z_ls[:int((n+1)*K),:]
        forward_z_train= forward_z_train[0:int(K*(n+1)-test_lag),:]

        forward_x_train= X_ls[:int((n+1)*K),:]
        forward_x_train= forward_x_train[test_lag:int(K*(n+1)),:]

        backward_z_train= Z_ls[:int((n+1)*K),:]
        backward_z_train= backward_z_train[1:int(K*(n+1)-test_lag+1),:]

        backward_x_train= X_ls[:int((n+1)*K),:]
        backward_x_train= backward_x_train[0:int(K*(n+1)-test_lag),:]

        z_test0=Z_ls[int((n+1)*K):int((n+2)*K),:]

        # forward dim 1
        note=0
        z_train = forward_z_train
        x_train0 = forward_x_train
        x_train=x_train0[:,[note]]

        hidden_units=config.h1; k_mixt=config.k1 ; num_iter=config.n1
        sample=MDN_learning1(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,lr)

        x_fake_ls1=tf.transpose(sample)
        x_fake_ls1=[tf.reshape(a,shape=[-1,1]) for a in x_fake_ls1]

        ######  dim 2 curve
        note=1
        z_train=forward_z_train
        z_train=np.concatenate((z_train,forward_x_train[:,:note]),1)
        x_train=forward_x_train[:,[note]]

        hidden_units=config.h2; k_mixt=config.k2; num_iter=config.n2
        sample=MDN_learning2(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,x_fake_ls1,lr)
        x_fake_ls2=tf.transpose(sample)
        x_fake_ls2=[tf.reshape(a,shape=[-1,1]) for a in x_fake_ls2]

        x_fake_ls12=[]
        for k in range(len(x_fake_ls1)):
            x_fake_ls12.append(tf.concat([x_fake_ls1[k],x_fake_ls2[k]],1))

        ######  dim 3 curve
        note=2
        z_train=forward_z_train
        z_train=np.concatenate((z_train,forward_x_train[:,:note]),1)
        x_train=forward_x_train[:,[note]]

        hidden_units=config.h3; k_mixt=config.k3; num_iter=config.n3
        sample=MDN_learning2(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,x_fake_ls12,lr)
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

        ######  dim 2 curve
        note=1
        z_train=backward_z_train
        z_train=np.concatenate((z_train,backward_x_train[:,:note]),1)
        x_train=backward_x_train[:,[note]]

        hidden_units=config.h5; k_mixt=config.k5; num_iter=config.n5
        sample=MDN_learning2(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,x_fake_back_ls1,lr)
        x_fake_back_ls2=tf.transpose(sample)
        x_fake_back_ls2=[tf.reshape(a,shape=[-1,1]) for a in x_fake_back_ls2]

        x_fake_back_ls12=[]
        for k in range(len(x_fake_back_ls1)):
            x_fake_back_ls12.append(tf.concat([x_fake_back_ls1[k],x_fake_back_ls2[k]],1))


        ######  dim 3 curve
        note=2
        z_train=backward_z_train
        z_train=np.concatenate((z_train,backward_x_train[:,:note]),1)
        x_train=backward_x_train[:,[note]]

        hidden_units=config.h6; k_mixt=config.k6; num_iter=config.n6
        sample=MDN_learning2(z_train,x_train,z_test0,hidden_units,k_mixt,M,num_iter,x_fake_back_ls12,lr)
        x_fake_back_ls3=tf.transpose(sample)
        x_fake_back_ls3=[tf.reshape(a,shape=[-1,1]) for a in x_fake_back_ls3]

        x_fake_back_ls123=[]
        for k in range(len(x_fake_back_ls1)):
            x_fake_back_ls123.append(tf.concat([x_fake_back_ls12[k],x_fake_back_ls3[k]],1))
        sim_backward=np.array(x_fake_back_ls123)

        r_ls=[]

        base=np.matmul(sim_forward,uv[0].T)
        r1=np.mean(np.cos(base),1)
        r2=np.mean(np.sin(base),1)
        r_ls.append([r1,r2])

        base=np.matmul(sim_backward,uv[1].T)
        r1=np.mean(np.cos(base),1)
        r2=np.mean(np.sin(base),1)
        r_ls.append([r1,r2])

        for i in range(2):  # forward / backward
            r = r_ls[i]
            char_values[0 + i][n] = r[0].reshape((1, K, B))
            char_values[2 + i][n] = r[1].reshape((1, K, B)) 

    series_ls_new=[]
    for n in range(N):
        series_ls_new.append(series_ls[int((n+1)*K):int((n+2)*K),:])

    estimated_cond_char = char_values
    series_array=np.array(series_ls_new)
    observed_cond_char = []
    for i in range(2):
        temp = series_array.dot(uv[i].T)
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
    S = S_hat(lam = lam, dims = [N, K], J = test_lag)  # Construct the test statistics
    pValues = bootstrap_p_value(Sigma_q_s, rep_times = int(1e3), test_stat = S) 
    return (pValues)