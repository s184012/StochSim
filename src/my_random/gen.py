import itertools
from re import I
from typing import Callable, Protocol
import numpy as np
from scipy.stats import uniform

def lcg(a, c, M, n, x=0):
    for _ in range(int(n)):
        x = (a*x + c) % M
        yield x / M

def geometric(p, size):
    u = uniform.rvs(size=size)
    return np.log(u) // np.log(1-p) + 1

def exponential(lmbda, size):
    u = uniform.rvs(size=size)
    return - np.log(u) / lmbda

def pareto(k , beta, size, loc=0):
    u = uniform.rvs(size=size)
    return beta*(u**(-1/k) - loc)

def norm_box_mueller(size):
    u1 = uniform.rvs(size=size)
    r = np.sqrt(-2*np.log(u1))
    return r*sin_cos(size)

def sin_cos(size):
    sin, cos = [], []
    while len(cos) < size / 2:
        v1, v2 = uniform.rvs(loc=-1, scale=2, size=2)
        r2 = v1**2 + v2**2
        if r2 <= 1:
            cos.append(v1/np.sqrt(r2))
            sin.append(v2/np.sqrt(r2))
    
    return np.array(sin + cos)


def discrete_crude(p, size):
    u =  uniform.rvs(size=size)
    probs = np.concatenate([[0], np.cumsum(p), [np.inf]], axis=0)
    x = np.zeros_like(u)
    for i, p in enumerate(probs):
        if 0 < i:
            x += ((probs[i-1] < u) & (u <= p)) * i

    return x

def discrete_rejection(p, size):
    c = max(p)
    k = len(p)
    I = []
    while len(I) < size:
        u1, u2 = uniform.rvs(size=2)
        i = int(np.floor(k*u1)) +1
        if u2 <= p[i-1]/c:
            I.append(i)
    
    return I

def discrete_alias(p, size):
    k = len(p)
    p = np.array(p)
    L = list(range(1,k+1))
    F = k*p
    G, S = np.where(F>=1)[0] + 1, np.where(F<=1)[0] + 1
    while len(S) > 0:
        i, j = int(G[0]), int(S[0])
        L[j-1] = i
        F[i-1] = F[i-1] - (1-F[j-1])
        if F[i-1] < 1:
            G = np.delete(G, 0)
            print(S)
            S = np.append(S, i)
            print(S)
        S = np.delete(S, 0)
        
    
    result = []
    while len(result) < size:
        u1, u2 = uniform.rvs(size=2)
        i = int(np.floor(k*u1)) + 1
        if u2 <= F[i-1]:
            result.append(i)
        else:
            result.append(L[i-1])
        
    return result


