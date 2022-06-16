from typing import Iterable, Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt


def group_count(p: Iterable, c: int) -> np.ndarray:
    split = 1 / c
    splits = np.zeros(c)
    for n in p:
        for i in range(c):
            if n <= split*(i+1):
                splits[i] += 1
                break
    return splits
            

def chi2(obs: np.ndarray, exp: np.ndarray, df=None):
    if df is None:
        df = len(obs) - 1
    T = sum((obs - exp)**2 / exp)
    p = 1 - stats.chi2.cdf(x=T, df=df)
    return p

def group_chi_test(obs, n_groups):
    splits = group_count(obs, n_groups)
    p = chi2(splits, np.ones_like(splits)*len(obs)/n_groups)
    return p

def emperical_dist(x: float, obs: np.ndarray) -> np.ndarray:
    return 1/len(obs) * sum(obs <= x)

def kolmogorov(obs, dist=stats.uniform, range=(0,1)):
    n = len(obs)
    obj = lambda x: - emperical_dist(x, obs) + dist.cdf(x)
    d_n = opt.minimize_scalar(obj, bounds = range, method='bounded').x
    return (np.sqrt(n) + 0.12 + 0.11/np.sqrt(n))*d_n, d_n

def runtest_above_below_median(obs: np.ndarray) -> int:
    T, n1, n2 = counts_above_and_below_median(obs)
    mean = (2*n1*n2)/(n1+n2) + 1
    var = (2*n1*n2*(2*n1*n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 -1))
    print(mean, var, T)
    T = (T - mean)/np.sqrt(var)
    
    return 2* (1- stats.norm.cdf(abs(T)))

def counts_above_and_below_median(obs: np.ndarray) -> Tuple[int, int, int]:
    r_a = obs > np.median(obs)
    r_b = obs < np.median(obs)
    x, count = 0, 0
    for a, b in zip(r_a, r_b) :
        if a==1 and x != 1:
            count += 1
            x = 1
        elif b==1 and x != -1:
            count += 1
            x = -1 

    return count, sum(r_a), sum(r_b)


def runtest_up_down_lengths(obs: np.ndarray) -> int:
    n = len(obs)
    r = run_lengths_increasing_count(obs)
    a = np.array(
        [
        [4529.4, 9044.9, 13568, 18091, 22615, 27892],
        [9044.9, 18097, 27139, 36187, 45234, 55789],
        [13568, 27139, 40721, 54281, 67852, 83685],
        [18091, 36187, 54281, 72414, 90470, 111580],
        [22615, 45234, 67852, 90470, 113262, 139476],
        [27892, 55789, 83685, 111580, 139476, 172860]
        ]
    )
    b = np.array([1/6, 5/24, 11/120, 19/720, 29/5040, 1/840])
    z = 1/(n-6) * ((r - n*b).T @ a @ (r - n*b))

    return 1 - stats.chi2.cdf(z, df=6)

def run_lengths_increasing_count(obs: np.ndarray) -> np.ndarray:
    prev_x = -np.inf
    r = np.zeros(6)
    count = 0
    for x in obs:
        if x <= prev_x:
            count = min(6, count)
            r[count-1] += 1
            prev_x = x
            count = 1
        else:
            prev_x = x
            count += 1
    
    count = min(6, count)
    r[count-1] += 1
    
    return r


def runtest_increase_decrease(obs):
    t = run_count_increase_decrease(obs)
    n = len(obs)
    z = (t - (2*n - 1)/3) / np.sqrt((16*n - 29)/90)
    return 2 * (1- stats.norm.cdf(abs(z)))



def run_count_increase_decrease(obs: np.ndarray):
    x, count, prev= 0, 0, obs[0]
    for y in obs[1:]:
        if x != 1 and y > prev:
            count += 1
            x = 1
        elif x != -1 and y <= prev:
            count += 1
            x = -1
        prev = y
    return count
    

def corr_est(obs, max_lag) -> np.ndarray:
    n = len(obs)
    c = np.zeros(max_lag)
    for lag in range(max_lag):
        low = obs[:n-lag-1]
        upp = obs[lag+1:]
        c[lag] = 1/(n-lag) * low @ upp

    return c

def plot_corr(obs: np.ndarray, max_lag=5, conf=0.05) -> None:
    n = len(obs)
    corr_coef = (corr_est(obs, max_lag) - 0.25)
    x = np.arange(1, len(corr_coef)+1)
    conf = stats.norm.ppf(1 - conf/2) * np.sqrt((7/(144*n)))
    plt.plot(x, corr_coef, 'ob')
    plt.vlines(x, np.zeros_like(x), corr_coef)
    plt.hlines([conf, 0, -conf], 0, max_lag+1, linestyles=['dashed', 'solid', 'dashed'])
    plt.show()
    

def all_test(obs, groups=100, lag=5, plot=True):
    p_chi = group_chi_test(obs, 100)
    T_kol = kolmogorov(obs)
    p_ab_median = runtest_above_below_median(obs)
    p_ud = runtest_up_down_lengths(obs)
    p_inc_dec = runtest_increase_decrease(obs)

    print(f'____________Uniform Distribution Tests___________')
    print(f'Chi^2 test with {groups} groups:                p={p_chi:.2f}')
    print(f'Kolmogorov Smirnof:                        T={T_kol:.2f}')
    print(f'_______________Independence Tests________________')
    print(f'Run Test 1: Above/below Median:            p={p_ab_median:.2f}')
    print(f'Run Test 2: Up/Down length count Test:     p={p_ud:.2f}')
    print(f'Run Test 3: Up/Down run count Test:        p={p_inc_dec:.2f}')
    if plot:
        plt.plot(obs[1:], obs[0:-1], '.')
        plt.show()
        plot_corr(obs)

    return p_chi, T_kol, p_ab_median, p_ud, p_inc_dec


if __name__ == '__main__':
    obs = stats.uniform.rvs(size=10_000)
    all_test(obs)