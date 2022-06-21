import numpy as np
from scipy import stats
import random

def gen_stations(n, min = 0, max = 200):
    stations = np.zeros((2,n))
    for i in range(n):
        stations[0][i]=random.randint(min,max)
        stations[1][i]=random.randint(min,max)
    return stations

def euclDist(a,b):
    dist=np.sqrt(np.power(b[0]-a[0],2)+np.power(b[1]-a[1],2))
    return dist