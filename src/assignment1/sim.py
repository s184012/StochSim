from dataclasses import dataclass
from scipy import stats
import numpy as np

P = np.zeros([5, 5])
P[0,:] =    [0.9915, 0.005, 0.0025, 0, 0.001]
P[1,1:] =   [0.986, 0.005, 0.004, 0.005]
P[2,2:] =   [0.992, 0.003, 0.005]
P[3,3:] =   [0.991, 0.009]
P[4,4:] =   1

@dataclass
class Woman:
    age: int
    stages: np.ndarray

    @property
    def local_reappearence(self):
        return np.any(self.stages == 1) or np.any(self.stages == 3)

    @property
    def death(self):
        return np.any(self.stages==4)

    def reappearence_within_first(self, months):
        return np.any((self.stages[:months+1] == 1) | (self.stages[:months+1] == 2))
    
    def stage_at(self, time):
        if time <= self.age:
            return self.stages[time] 
        else:
            return 4


@dataclass
class Population:
    women: 'list[Woman]'

    @property
    def size(self):
        return len(self.women)

    @property
    def local_reappearence_fraction(self):
        return sum(woman.local_reappearence for woman in self.women) / self.size

    @property
    def ages(self) -> np.ndarray:
        return np.array([woman.age for woman in self.women])
    

    def death_fraction_at(self, time):
        return self.women_death_at(time) / self.size

    
    def mean_age(self, alpha):
        mu = self.ages.mean()
        sd = self.ages.std()
        t = np.array(stats.t.interval(alpha, df=self.size-1))
        return mu, mu + t*(sd/np.sqrt(self.size))
    
    def women_death_at(self, time):
        return sum(woman.death for woman in self.women if woman.stage_at(time) == 4)

    def stage_distribution(self, time):
        dist = []
        for stage in range(5):
            count = sum(woman.stage_at(time) == stage for woman in self.women)
            dist.append(count)
        
        return dist


def p_utils():
    Ps = P[:-1, :-1]
    p0 = expected_stage_distribution_at(0)[:-1]
    ps = P[:-1, -1]
    return Ps, p0, ps

def lifetime_expectation():
    Ps, p0, _ = p_utils()
    return p0 @ np.linalg.inv(np.eye(4) - Ps) @ np.ones(4)


def lifetime_pdf(start, end):
    Ps, p0, ps = p_utils()
    Ps_i = np.linalg.matrix_power(Ps,start)
    f = []
    for t in range(start, end+1):
        f.append(p0 @ Ps_i @ ps)
        Ps_i = Ps_i @ Ps
    
    return f


def lifetime_cdf(times):
    return [sum(lifetime_pdf(0, t)) for t in times]

def expected_stage_distribution_at(time):
    p0 = np.array([1, 0, 0, 0, 0])
    return  p0 @ np.linalg.matrix_power(P, time)

def check_distribution_at(pop: Population, time):
    pt = expected_stage_distribution_at(time)
    return stats.chisquare(pop.stage_distribution(time), pt*pop.size)


def p(stage):
    return P[stage,:]

def sim_woman(stage=0) -> Woman:
    lifetime = 0
    stages = [stage]
    while stage < 4:
        stage = np.random.choice(5, p=p(stage))
        stages.append(stage)
        lifetime += 1
    
    return Woman(lifetime, np.array(stages))

def sim_population(size, start_state=0) -> Population:
    women = [sim_woman(start_state) for _ in range(size)] 
    return Population(women)


def sim_population_with_criteria(size, months):
    women = []
    while len(women) < size:
        woman = sim_woman()
        if woman.age >= months and woman.reappearence_within_first(months):
            women.append(woman)
        if len(women) % 200 == 0:
            print(len(women))
    
    return Population(women)


