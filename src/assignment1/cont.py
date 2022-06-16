from dataclasses import dataclass, field
from enum import Enum
from tracemalloc import start
import numpy as np
from scipy import stats
from scipy.linalg import expm

Q = np.zeros([5,5])
Q[0,:] =    [-0.0085, 0.005, 0.0025, 0, 0.001]
Q[1, 1:] =  [-0.014, 0.005, 0.004, 0.005]
Q[2, 2:] =  [-0.008, 0.003, 0.005]
Q[3, 3:] =  [-0.009, 0.009]

Q9 = np.zeros((5,5))
Q9[0, 1:] = [0.0025, 0.00125, 0, 0.001]
Q9[1, 2:] = [0, 0.002, 0.005]
Q9[2, 3:] = [0.003, 0.005]
Q9[3, 4:] = [0.009]
np.fill_diagonal(Q9, -np.sum(Q9, axis=1))



class State(Enum):
    S1 = 0
    S2 = 1
    S3 = 2
    S4 = 3
    DEAD = 4


@dataclass
class ContWoman:
    lifetime: float 
    state_timeline: dict
    state_time: dict = field(init=False)

    def __post_init__(self):
        self.state_time = {state: time for time, state in self.state_timeline.items()}

    @property
    def last_state(self):
        return self.states[-1]

    @property
    def states(self):
        return list(self.state_timeline.values())

    @property
    def had_distant_reappearance(self):
        return self.had_state(State.S3) or self.had_state(State.S4)
    
    def state_at(self, time):
        prev_state = State.S1
        for t, curr_state in self.state_timeline.items():
            if t > time:
                return prev_state
            prev_state = curr_state
        
        return State.DEAD
            

    def observed_states(self, freq=48):
        result = []
        for t in range(np.ceil(self.lifetime).astype(int) + 1):
            if t % 48 == 0:
                result.append(self.state_at(t))
        
        if result[-1] != State.DEAD:
            result.append(State.DEAD)
        return result
            
        

    def had_state(self, state: State) -> bool:
        return state in self.states
    

    
@dataclass
class ContPopulation:
    women: 'list[ContWoman]'

    @property
    def size(self):
        return len(self.women)
    
    @property
    def lifetimes(self) -> np.ndarray:
        return np.array([woman.lifetime for woman in self.women])
    
    def mean_lifetime(self, alpha):
        mu = self.lifetimes.mean()
        sd = self.lifetimes.std()
        t = np.array(stats.t.interval(alpha, df=self.size-1))
        return mu, mu + t*(sd/np.sqrt(self.size))
    
    def sd_lifetime(self, alpha):
        std = self.lifetimes.std()
        sd = bootstrap(self.lifetimes, np.std, 10_0000)
        return std, stats.norm(loc=std, scale=sd).interval(alpha)




    def number_of_deaths_at_time(self, time):
        return np.count_nonzero(self.lifetimes <= time)


    def women_with_state(self, state: State):
        return [woman for woman in self.women if woman.had_state(state)]

    def women_with_state_after_time(self, state: State, time):
        result = []
        for woman in self.women:
            states = {state for t, state in woman.state_timeline.items() if t > time}
            if state in states:
                result.append(woman)
        
        return result

    def women_with_distant_reappaerance(self):
        return [woman for woman in self.women if woman.had_distant_reappearance]

    def women_with_distant_reappaerance_after_time(self, time):
        reapp_time = lambda woman: max(woman.state_time.get(State.S3, 0), woman.state_time.get(State.S4, 0)) 
        return [woman for woman in self.women_with_distant_reappaerance() if reapp_time(woman) > time]

    def state_proportion(self, state: State):
        return len(self.women_with_state(state)) / self.size

    def state_proportion_after_time(self, state: State, time):
        return len(self.women_with_state_after_time(state, time)) / self.size

    def distant_reappearance_proportion_after_time(self, time):
        return len(self.women_with_distant_reappaerance_after_time(time))/self.size

    def emperical_lifetime_distribution(self, time_interval):
        return [1/self.size * sum(self.lifetimes <= t) for t in range(*time_interval)]
    
    def survival_function(self, times):
        return [(self.size - self.number_of_deaths_at_time(t)) / self.size for t in times]

    def observed_population(self, period=48):
        return [woman.observed_states() for woman in self.women]


def QS(Q=Q):
    return Q[:-1, :-1]

def sim_stay_time(state: State, Q = Q):
    return stats.expon(scale=1/-Q[state.value, state.value]).rvs()

def switch_probability(state: State, Q = Q):
    p = -Q[state.value, :]/Q[state.value, state.value]
    p[p==-1] = 0
    return p

def sim_next_state(state: State = State.S1, Q=Q):
    return np.random.choice(list(State), p=switch_probability(state, Q=Q))

def sim_woman(start_state: State = State.S1, Q=Q):
    t, state_timeline = 0, {0: start_state}
    while state_timeline[t] != State.DEAD:
        next_state = sim_next_state(state_timeline[t], Q)
        t += sim_stay_time(state_timeline[t], Q)
        state_timeline[t] = next_state
    return ContWoman(t, state_timeline)

def sim_population(size, start_state: State = State.S1, Q=Q):
    women = [sim_woman(start_state, Q=Q) for _ in range(size)]
    return ContPopulation(women)

def true_lifetime_distribution(times, start_state: State = State.S1, Q=Q):
    p0 = np.zeros(len(State) - 1)
    p0[start_state.value] = 1
    return [1 - sum(p0 @ expm(QS(Q)*t)) for t in times]

def sim_woman_until(start_time, start_state, period, Q):
    t, times, states = start_time, [start_time], [start_state]

    while states[-1] != State.DEAD:
        t += sim_stay_time(states[-1], Q)
        next_state = sim_next_state(states[-1], Q)
        if t > start_time + period:
            break
        states.append(next_state)
        times.append(t)
    return states[-1], dict(zip(times, states))


def sim_woman_from_obs(observations, Q, period=48):
    t, state, state_timeline = 0, State.S1, {0: State.S1}
    i = 0
    while i < len(observations):
        last_state, timeline = sim_woman_until(t, state, period, Q)
        if last_state.value == observations[i].value:
            i += 1
            state_timeline.update(timeline)
            t += period
            state = last_state
    return state_timeline


def get_Q(observations, Q):
    N = np.zeros((4,5))
    S = np.zeros(len(State)-1)
    for it, obs in enumerate(observations):
        traj = sim_woman_from_obs(obs, Q=Q)
        i = 0
        for time, state in traj.items():
            if state.value != i:
                N[i, state.value] += 1
                S[state.value-1] += time
                i = state.value
    
    new_Q = np.zeros((5,5))
    new_Q[:4, :] = (N.T/S).T
    np.fill_diagonal(new_Q, -np.sum(new_Q, axis=1))
    return new_Q

def infer_Q_from_observations(obs, Q0, tol=1e-3):
    Q1 = np.inf
    while np.max(np.abs(Q0 - Q1)) > tol:
        Q1 = Q0
        Q0 = get_Q(obs, Q0)
    
    return Q0

def bootstrap(data, stat_func=lambda x: np.median, size = 1000):
    X = [np.random.choice(data, len(data)) for _ in range(size)]
    stat = stat_func(X, axis=1)
    return stat.std()

if __name__ == '__main__':
    pass

