import numpy as np
import ipdb
import sys
sys.path.append("./utils/")
from utils_functions import grad_l

class OnlinePlayer(object):
    """
    Player that implements the Adaptive Gradient Play algorithm
    """

    def __init__(self, d_i, d, m_i, eta, nu, n, B, R):
        self.theta_history = []
        self.theta_other_history = []
        self.data_history = []
        self.risk_history = []
        self.p_hat_history = []
        self.q_history = []
        self.u_history = []
        self.u_other_history = []

        self.x_train = None
        self.qs = None
        
        self.d = d
        self.d_i = d_i
        self.m_i = m_i
        self.eta = eta
        self.nu = nu
        self.n = n
        self.B = B
        self.R = R
        
        self.initialize_p_hat()

    def initialize_theta(self, d=0):
        if len(self.theta_history) != 0:
            raise ValueError("theta has already been initialized")
        theta_init = np.zeros((self.d_i,1))
        self.theta_history.append(theta_init)
        return theta_init

    def initialize_p_hat(self):
        if len(self.p_hat_history) != 0:
            raise ValueError("p_hat has already been initialized")
        p_hat_init = np.zeros((1,self.d))
        self.p_hat_history.append(p_hat_init)
        return p_hat_init
    
    def perturb_theta(self):
        theta_t = self.theta_history[-1]
        u_k = self.B*np.random.randn(self.d_i,self.n)
        self.u_history.append(u_k)
        return theta_t, u_k
    
    def compute_aux_q(self, z_k, theta_t_other, u_k_other):
        self.data_history.append(np.array(z_k).reshape((self.m_i,-1)))
        self.theta_other_history.append(theta_t_other)
        self.u_other_history.append(u_k_other)
        theta_other_k = theta_t_other+u_k_other
        p_hat_t = self.p_hat_history[-1]
        theta_k = self.theta_history[-1]+self.u_history[-1]
        q_k = np.array(z_k).reshape((self.m_i,-1))-np.dot(p_hat_t,np.vstack((theta_k,theta_other_k)))
        self.q_history.append(q_k)
        return q_k
    
    def update_p(self):
        p_hat_old = self.p_hat_history[-1]
#         print('phat',p_hat_old.shape)
        mu_hat = p_hat_old[:,:self.d_i]
#         print('mu',mu_hat.shape)
        gamma_hat = p_hat_old[:,self.d_i:]
#         print('gamma',gamma_hat.shape)
        z_t = self.data_history[-1]
#         print('zt',z_t.shape)
        theta_me_t = (self.theta_history[-1]+self.u_history[-1]).reshape((self.d_i,-1))
#         print('thetame',theta_me_t.shape)
        theta_other_t = (self.theta_other_history[-1]+self.u_other_history[-1]).reshape((self.d-self.d_i,-1))
#         print('thetaother',theta_other_t.shape)
        theta_t = np.vstack((theta_me_t,theta_other_t))
#         print('theta',theta_t.shape)
#         print(np.dot(z_t - np.dot(np.hstack((np.diagflat(self.theta_history[-1]),np.diagflat(self.theta_other_history[-1]))),theta_t),theta_t.T))
        mu_step = np.diag(np.dot(z_t - np.dot(np.hstack((np.diagflat(mu_hat),np.diagflat(gamma_hat))),theta_t),theta_t.T)[:,:self.d_i]).reshape((1,-1))
        gamma_step = np.diag(np.dot(z_t - np.dot(np.hstack((np.diagflat(mu_hat),np.diagflat(gamma_hat))),theta_t),theta_t.T)[:,self.d_i:]).reshape((1,-1))
        p_hat_new = np.clip(p_hat_old + (self.nu/self.n)*np.hstack((mu_step,gamma_step)), -self.R, self.R)
#         print('phatnew', p_hat_new, p_hat_new.shape)
        self.p_hat_history.append(p_hat_new)
        return p_hat_new
    
    def update_theta(self, oracle_grad):
        theta_me_old = self.theta_history[-1]
        theta_other_old = self.theta_other_history[-1]
        theta_t = np.vstack((theta_me_old,theta_other_old))
        q_k = np.array(self.q_history[-1]).reshape((self.m_i,-1))
        p_t = self.p_hat_history[-1]
        theta_me_new = np.clip(theta_me_old - (self.eta/self.n)*np.sum(oracle_grad, axis=1).reshape((-1,1)), -self.R, self.R)
        self.theta_history.append(theta_me_new)
        return theta_me_new