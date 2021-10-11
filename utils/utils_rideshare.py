import numpy as np
import ipdb

"""
Data
"""
def sample_from_location_family_rideshare(g, mu, gamma, theta_me, theta_other, n=1):
    y = np.random.poisson(lam=g, size=(len(g),n)) + np.multiply(mu,theta_me) + np.multiply(gamma,theta_other)
    return y.reshape(-1,n)

def sample_from_location_family_toy(g, mu, gamma, theta_me, theta_other, n=1):
    y = np.random.normal(loc=g, scale=1, size=(len(g),n)) + np.multiply(mu,theta_me) + np.multiply(gamma,theta_other)
    return y.reshape(-1,n)

def location_family_demand_rideshare(g, mu, gamma, theta_me, theta_other):
    y = g + np.multiply(mu,theta_me) + np.multiply(gamma,theta_other)
    return np.maximum(y,np.zeros(y.shape)).reshape(-1,1)

def sample_sphere(epsilon,d):
    """
    Returns a point on the sphere in R^d of radius epsilon
    """
    x = np.random.normal(size=(d,1))
    x /= np.linalg.norm(x)
    x *= epsilon
    return x


"""
Evaluate PR
"""

def evaluate_performative_risk(demand_generator, g, prices_, mu, gamma, lambda_r,
                               theta_me, theta_other, num_test):
    risk = np.mean(np.dot(-demand_generator(g, mu, gamma, theta_me, theta_other, num_test).T,(theta_me+prices_))+(lambda_r/2*(np.linalg.norm(theta_me)**2)))
    return risk


"""
Helpers for TwoStage Player
"""

def solve_distribution_params(z_lst, theta_me_lst, theta_other_lst):
    mu_hat = []
    gamma_hat = []
    for i in range(np.size(z_lst, axis=1)):
        y = np.array([e[i] for e in z_lst])
        A = np.hstack((np.array(theta_me_lst)[:,i], np.array(theta_other_lst)[:,i], np.ones((np.size(theta_me_lst, axis=0), 1))))
        mu_tilde = np.dot(np.dot(np.linalg.pinv(np.dot(A.T, A)), A.T), y)
        mu_hat.append(mu_tilde[0])
        gamma_hat.append(mu_tilde[1])
    return np.array(mu_hat).reshape((-1,1)), np.array(gamma_hat).reshape((-1,1))

def find_qs(mu_hat, gamma_hat, z_lst, theta_me_lst, theta_other_lst):
    y_lst = [e for e in z_lst]
    q_lst = []
    for (idx, y) in enumerate(y_lst):
        theta_me = theta_me_lst[idx]
        theta_other = theta_other_lst[idx]
        q = y - np.dot(mu_hat.T, theta_me) - np.dot(gamma_hat.T, theta_other)
        q_lst.append(q)
    return np.array(q_lst)

def solve_theta(g, prices_, eta, lambda_r, mu_hat, gamma_hat, theta_me, theta_other):
    theta = np.linalg.inv(lambda_r*np.eye(theta_me.shape[0])-2*np.diag(mu_hat))@(np.diag(mu_hat)@prices_ + g + np.diag(gamma_hat)@theta_other)
    return theta

def gd_theta(g, prices_, eta, lambda_r, t, mu_hat, gamma_hat, theta_me, theta_other):
    theta = theta_me - (eta/np.log10(t+2))*(-g - np.multiply(mu_hat,prices_) - (2*np.multiply(mu_hat,theta_me)) - (np.multiply(gamma_hat,theta_other)) + (lambda_r*theta_me))
    return theta

def solve_mu(z_lst, theta_me_lst):
    y = [e[1] for e in z_lst]
    A = np.array(theta_me_lst)
    mu_hat = np.linalg.pinv(A.T @ A) @ A.T @ y
    return mu_hat