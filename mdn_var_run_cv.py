from markov_test.MDN_VAR_CV import *
import argparse

parser = argparse.ArgumentParser(description='mdn')
parser.add_argument('-T', '--T', type=int, default=500)
parser.add_argument('-test_lag', '--test_lag', type=int, default=3)
parser.add_argument('-L', '--L', type=int, default=3)
parser.add_argument('-B', '--B', type=int, default=1000)
parser.add_argument('-M', '--M', type=int, default=100)
parser.add_argument('-Q', '--Q', type=int, default=10)
parser.add_argument('-lr', '--lr', type=float, default=0.005)
parser.add_argument('-num_h', '--num_h', type=int, default=20)
parser.add_argument('-n_iter', '--n_iter', type=int, default=6000)
args0 = parser.parse_args()

string = "mdn_var_run_cv" + "_T_" + str(args0.T) + "_test_lag_" + str(args0.test_lag)+ "_L_" + str(args0.L)+ "_B_" + str(args0.B) + "_M_" + str(args0.M) + "_Q_" + str(args0.Q) + "_lr_" + str(args0.lr) + "_num_h_" + str(args0.num_h) + "_n_iter_" + str(args0.n_iter) 

class Setting:
    def __init__(self):
        self.T = args0.T
        self.x_dims=3
        self.nstd=0.5
        self.test_lag=args0.test_lag
        self.L=args0.L
        self.B=args0.B
        self.M=args0.M
        self.first_T=1000
        self.Q=args0.Q
        self.A1=np.array([[0.5,-0.2,-0.2],[-0.2,0.5,-0.2],[-0.2,-0.2,0.5]])
        self.A2=np.array([[-0.5,0.2,0.2],[0.2,-0.5,0.2],[0.2,0.2,-0.5]])
        self.A3=np.array([[0.4,-0.1,-0.1],[-0.1,0.4,-0.1],[-0.1,-0.1,0.4]])
        self.B1=np.array([[0.3,-0.1,-0.1],[-0.1,0.3,-0.1],[-0.1,-0.1,0.3]])
        self.B2=np.array([[-0.3,0.1,0.1],[0.1,-0.3,0.1],[0.1,0.1,-0.3]])
        self.B3=np.array([[0.3,-0.1,-0.1],[-0.1,0.3,-0.1],[-0.1,-0.1,0.3]])
        self.sim_type="var"
        self.lr=args0.lr
        self.constant=0.1
        self.show=False
        self.h1=args0.num_h; self.k1=None; self.n1=args0.n_iter
        self.h2=args0.num_h; self.k2=None; self.n2=args0.n_iter
        self.h3=args0.num_h; self.k3=None; self.n3=args0.n_iter
        self.h4=args0.num_h; self.k4=None; self.n4=args0.n_iter
        self.h5=args0.num_h; self.k5=None; self.n5=args0.n_iter
        self.h6=args0.num_h; self.k6=None; self.n6=args0.n_iter
        self.cv_numk = [1,3,5,7]
        
config=Setting()
if config.sim_type=="var":
    check_stationarity(config)

pvalue_ls=[]
for seed in range(500):
    print("seed:",seed)
    k_forward, k_backward = simulate_cv(seed,config)
    config.k1 = k_forward; config.k2 = k_forward; config.k3 = k_forward
    config.k4 = k_backward; config.k5 = k_backward; config.k6 = k_backward; 
    pvalue_ls.append([simulate(seed,config),k_forward,k_backward])
    np.save("result/"+string,pvalue_ls)