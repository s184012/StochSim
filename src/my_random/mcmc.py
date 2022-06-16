from itertools import product
from math import factorial, pi
import numpy as np
from scipy.stats import binom, uniform, multivariate_normal, norm
import random

def h_1(x, y, m=10):
    return .5 if abs(x-y) % (m-1) == 1 else 0

def step_1(x, m=10):
    dx = 1 if binom.rvs(1, .5) == 1 else -1
    return (x + dx) % (m+1)

def g_1(x, a=8, m=10):
    return a**x / factorial(x)

def mcmc_1(x0, g, h, step, a=8, m=10, size = 10_000, burn_in = 100):
    x = x0
    for i in range(burn_in):
        y = step_1(x, m)
        cond = (g(y) * h(y,x)) / (g(x)*h(x,y))
        if uniform.rvs() <= cond:
            x = y
    
    states = []
    for i in range(size):
        y = step_1(x, m)
        cond = (g(y) * h(y,x)) / (g(x)*h(x,y))
        if uniform.rvs() <= cond:
            x = y
        states.append(x)

    return states

def set_of_valid_points(m=10):
    point_in_set = lambda i,j: 0 <= i + j <= m \
        and i >= 0 \
        and j >=0
    return {(i,j) for i,j in product(range(m+1), repeat=2) if point_in_set(i,j)}

def nearby_points(x, m=10):
    for i,j in product([-1, 0, 1], repeat=2):
        if i == j:
            continue
        new_point = (x[0] + i, x[1] + j)
        if new_point in set_of_valid_points(m):
            yield new_point

def cardinal_points(x, m=10):
    for i,j in product([-1, 0, 1], repeat=2):
        if i == j:
            continue
        if i!=0 and j!=0:
            continue
        
        new_point = (x[0] + i, x[1] + j)
        if new_point in set_of_valid_points(m):
            yield new_point

def g2(x, m=10, a1=4, a2=4):
    return a1**x[0]* a2**x[1] / (factorial(x[0])*factorial(x[1]))


def h2a(x, y, m=10):
    if y[0] + y[1] > m:
        return 0
    valid_count = 0
    for p in nearby_points(x, m):
        valid_count+=1
    
    
    return 1/valid_count


def step2a(x, m):
    return random.choice([p for p in nearby_points(x, m)])
    

def h2b(x, y, m=10):
    if y[0] + y[1] > m:
        return 0
    
    valid_count = 0
    for p in cardinal_points(x, m):
        valid_count += 1

    return 1/valid_count


def step2b(x, m):
    return random.choice([p for p in cardinal_points(x, m)])



def mcmc(x0, g, h, step, a=8, m=10, size = 10_000, burn_in = 100):
    x = x0
    for _ in range(burn_in):
        y = step(x, m)
        cond = (g(y) * h(y,x)) / (g(x)*h(x,y))
        if uniform.rvs() <= cond:
            x = y
    
    states = []
    for _ in range(size):
        y = step(x, m)
        cond = (g(y) * h(y,x)) / (g(x)*h(x,y))
        if uniform.rvs() <= cond:
            x = y
        states.append(x)

    return states


def p2(i, j, m=10):
    return g2((i,j)) / sum([g2((i,j)) for i,j in set_of_valid_points(m=10)])

def get_marginal_g2(i, x, m=10):
    j = x[(i+1) % 2]
    return [p2(i,j) / sum(p2(k,j) for k in range(m-j+1)) for i in range(m-j+1)]


def gibbs2c(x0, m = 10, size=10_000, burn=100):
    x = x0
    res = []
    for iter in range(size+burn):
        for i, x_i in enumerate(x):
            dist = get_marginal_g2(i, x, m)
            x[i] = np.random.choice(len(dist), p = dist)
        if iter >= burn:
            res.append(x.copy())
        
        if iter % 1000 == 0:
            print(iter)
    
    return res


def gen_xi_gamma(size=1):
    return multivariate_normal([0, 0], np.array([[1, .5],[.5, 1]])).rvs(size=size)

def gen_theta_psi(size=1):
    return np.exp(gen_xi_gamma(size=size))

def gen_observations(size=1):
    mean, var = gen_theta_psi()
    return norm(mean, np.sqrt(var)).rvs(size=size), (mean, var)

def norm_step(x):
    dx = norm(loc = 0, scale=1e-4).rvs(2)
    return x + dx

def g3(x, obs):
    ln_pdf = 1/(2*pi*x[0]*x[1]*np.sqrt(1 - .5**2))\
        *np.exp(- (np.log(x[0])**2 - np.log(x[0])*np.log(x[1]) + np.log(x[1])**2) \
            / 2*(1-.5**2))
    return sum(norm(loc=x[0], scale=np.sqrt(x[1])).pdf(obs)) * ln_pdf


def mcmc_continuous(x0, obs, g, step, burn_in=100, size=10_000):
    x = x0
    for _ in range(burn_in):

        y = step(x)
        cond = (g(np.exp(y), obs) / (g(np.exp(x), obs)))
        if uniform.rvs() <= cond:
            x = y
    
    states = []
    for i in range(size):
        y = step(x)
        cond = g(np.exp(y), obs) / g(np.exp(x), obs)
        if uniform.rvs() <= cond:
            x = y
        states.append(np.exp(x))
        if i % 1000 == 0:
            print(i)

    return states
    

if __name__ == '__main__':
    obs = gen_observations(10)
    print(mcmc_continuous([0,0], obs, g3, norm_step, size=10_000))

