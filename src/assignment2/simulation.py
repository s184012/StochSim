from dataclasses import dataclass, field
from enum import IntEnum, Enum, auto
import heapq
from typing import Callable, TypedDict, Union
import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd

sns.set_theme(style='darkgrid')
from src.assignment1.cont import switch_probability

class WardType(IntEnum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5

class PatientState(Enum):
    PENDING = 0
    IN_CORRECT_WARD = 1
    IN_WRONG_WARD = 2
    REJECTED = 3


@dataclass
class WardConfig:
    bed_capacity: int= 0
    urgency: int= 0
    mean_arrival_time: float= 0
    mean_stay_time: float= 0
    urgency: int= 0

class SwitchConfigurations(TypedDict):
    ward_type: WardType
    probability: 'list[float]'
    
@dataclass
class WardsConfigurations:
    configs: 'list[WardConfig]'
    def ward(self, ward: WardType):
        return self.configs[ward]
    
    def __getitem__(self, item):
        return self.configs[item]

    @property
    def bed_distribution(self):
        return [c.bed_capacity for c in self.configs]
    
    @bed_distribution.setter
    def bed_distribution(self, new_dist):
        for c, val in zip(self.configs, new_dist):
            c.bed_capacity = val
    
    @property
    def mean_arrival_times(self):
        return [c.mean_arrival_time for c in self.configs]
    
    @mean_arrival_times.setter
    def mean_arrival_times(self, new_times):
        for c, time in zip(self.configs, new_times):
            c.mean_arrival_time = time
    
    @property
    def mean_stay_times(self):
        return [c.mean_arrival_time for c in self.configs]
    
    @mean_stay_times.setter
    def mean_stay_times(self, new_times):
        for c, time in zip(self.configs, new_times):
            c.mean_arrival_time = time
    

meantime_paramterized_distribution = Callable[[float], float]

@dataclass
class HospitalConfiguration:
    wards_config: WardsConfigurations
    switch_config: SwitchConfigurations
    inter_arrival_time_distribution: meantime_paramterized_distribution
    stay_time_distribution: meantime_paramterized_distribution
    ward_list: 'list[WardType]' = WardType


@dataclass(order=True)
class Patient:
    preferred_ward: WardType=field(compare = False)
    arrival_time: float=field(repr=False)
    stay_time: dict=field(compare = False, repr=False)
    state: PatientState=field(compare = False, default=PatientState.PENDING)
    assigned_ward: WardType = None

    def update_ward(self, ward: WardType):
        self.assigned_ward = ward
        if self.assigned_ward is self.preferred_ward:
            self.state = PatientState.IN_CORRECT_WARD
        else:
            self.state = PatientState.IN_WRONG_WARD


@dataclass
class Ward:
    type: WardType
    config: WardConfig
    total_number_of_treated_patients: int = 0
    total_number_of_rejected_patients: int = 0
    patients: list = field(default_factory=list)
    
    def __post_init__(self):
        self.capacity = self.config.bed_capacity
        self.urgency = self.config.urgency

    @property
    def number_of_current_patients(self):
        """Number of currently enrolled patients"""
        return len(self.patients)

    @property
    def is_full(self):
        """Checks if ward is full"""
        return self.number_of_current_patients == self.capacity
    
    @property
    def is_empty(self):
        return self.number_of_current_patients == 0 or self.capacity == 0
    
    @property
    def number_of_processed_patients(self):
        """Total number of patients who have tried to get into the ward"""
        return self.total_number_of_treated_patients + self.total_number_of_rejected_patients

    @property
    def rejected_fraction(self):
        """Fraction of all processed patients who were rejected"""
        return self.total_number_of_rejected_patients / self.number_of_processed_patients
    
    @property
    def fill_fraction(self):
        if self.capacity == 0:
            return 1
        else:
            return self.number_of_current_patients / self.capacity
    
    @property
    def rellocation_score(self):
        """Score expressing how well the ward can handle a rellocation of a bed"""
        return self.fill_fraction * self.urgency/29

    
    def accepted_fraction(self, next_is_rejected=0):
        if self.number_of_processed_patients == 0:
            return 0
        return self.total_number_of_treated_patients / (self.number_of_processed_patients + next_is_rejected)
    

    def add_patient(self, patient: Patient):
        heapq.heappush(self.patients, (patient.arrival_time + patient.stay_time, patient))
        self.total_number_of_treated_patients += 1
        patient.update_ward(self.type)

    def reject_patient(self, patient: Patient):
        patient.state = PatientState.REJECTED
        self.total_number_of_rejected_patients += 1

    def process_patient(self, patient: Patient):
        if self.is_full:
            self.reject_patient(patient)
            return patient
        else:
            self.add_patient(patient)
            return None
    
    def update_patients(self, time):
        while self.patients and self.patients[0][0] <= time:
            heapq.heappop(self.patients)



class SimulationResult:
    """Contains the result of a simulation"""
    def __init__(self, patients: 'list[Patient]', penalty, ward_config: WardsConfigurations) -> None:
        self.patients = patients
        self.penalty = penalty
        self.config = ward_config
        self.number_of_patients = len(patients)
    
    def state(self, states: 'list[PatientState]'):
        return [p for p in self.patients if p.state in states]

    def state_pct_total(self, states: 'list[PatientState]'):
        count = len([p for p in self.patients if p.state in states])
        return count / self.number_of_patients * 100

    def in_ward(self, wards: 'list[WardType]'):
        return [p for p in self.patients if p.assigned_ward in wards]

    def from_ward(self, wards: 'list[WardType]'):
        return [p for p in self.patients if p.preferred_ward in wards]
    
    def states_in_wards(self, states: 'list[PatientState]', wards: 'list[WardType]'):
        wards.append(None)
        return [p for p in self.patients if p.state in states and p.assigned_ward in wards]
    
    def states_in_wards_pct(self, states: 'list[PatientState]', wards: 'list[WardType]'):
        return len(self.states_in_wards(states, wards)) / len(self.in_ward(wards)) * 100
    
    def states_from_wards(self, states, wards):
        return [p for p in self.patients if p.preferred_ward in wards and p.state in states]
    
    def states_from_wards_pct(self, states, wards):
        return len(self.states_from_wards(states, wards)) / len(self.from_ward(wards))
    
    def penalty_from_wards(self, ward: WardType):
        return len(self.states_from_wards([PatientState.REJECTED, PatientState.IN_WRONG_WARD], [ward])) * self.config[ward].urgency
    

def bootstrap(data, stat_func=lambda x: np.median, size = 1000):
    x = [np.random.choice(data, len(data)) for _ in range(size)]
    stat = stat_func(x, axis=1)
    return stat.std()

def mean_conf(mean, std, alpha=.95):
    t = stats.t.interval(alpha=alpha)



class SimulationsSummary:
    """Summarises and compares different simulation results"""
    def __init__(self, results: 'list[SimulationResult]'):
        self.results: 'list[SimulationResult]' = results
    
    def expected_probability_of_occupied(self, wards=list(WardType), alpha=0.95):
        means, lwrs, uprs = [], [], []
        for ward in wards:
            _, mean, (lwr, upr) = self.state_from_ward_pct_distribution([PatientState.IN_WRONG_WARD, PatientState.REJECTED], [ward], alpha)
            means.append(mean)
            lwrs.append(lwr)
            uprs.append(upr)
        index = [ward.name for ward in wards]
        return pd.DataFrame({'mean': means, 'lwr': lwrs, 'upr': uprs}, index=index).copy()
    
    def expected_total_penalty(self, alpha=0.95):
        dist = [r.penalty for r in self.results]
        mean = np.mean(dist)
        sd = bootstrap(dist, np.std, 10_000)
        return dist, mean, stats.norm(loc=mean, scale=sd).interval(alpha)      

    def state_from_ward_pct_distribution(self, states: 'list[PatientState]', wards: 'list[WardType]', alpha=0.95):
        dist = [r.states_from_wards_pct(states, wards) for r in self.results]
        mean = np.mean(dist)
        sd = bootstrap(dist, np.std, 10_000)
        return dist, mean, stats.norm(loc=mean, scale=sd).interval(alpha)
    
    def number_of_state_in_ward(self, states: 'list[PatientState]', wards: 'list[WardType]', alpha=0.95):
        dist = [len(r.states_in_wards(states, wards)) for r in self.results]
        mean = np.mean(dist)
        sd = bootstrap(dist, np.std, 10_000)
        return dist, mean, stats.norm(loc=mean, scale=sd).interval(alpha)
    
    def number_of_state_from_ward(self, states: 'list[PatientState]', wards: 'list[WardType]', alpha = 0.95):
        dist = [len(r.states_from_wards(states, wards)) for r in self.results]
        mean = np.mean(dist)
        sd = bootstrap(dist, np.std, 10_000)
        return dist, mean, stats.norm(loc=mean, scale=sd).interval(alpha)
    
    def penalty_from_wards(self, wards: 'list[WardType]', alpha=.95):
        dist = [r.penalty_from_wards(wards) for r in self.results]
        mean = np.mean(dist)
        sd = bootstrap(dist, np.std, 10_000)
        return dist, mean, stats.norm(loc=mean, scale=sd).interval(alpha)      

    def expected_admissions(self, wards: 'list[WardType]'=list(WardType)):
        means, lwrs, uprs = [], [], []
        for ward in wards:
            _, mean, (lwr, upr) = self.number_of_state_in_ward([PatientState.IN_CORRECT_WARD], [ward])
            means.append(mean)
            lwrs.append(lwr)
            uprs.append(upr)
        index = [ward.name for ward in wards]
        return pd.DataFrame({'mean': means, 'lwr': lwrs, 'upr': uprs}, index=index).copy()
    
    def expected_relocations(self, wards: 'list[WardType]'=list(WardType)):
        means, lwrs, uprs = [], [], []
        for ward in wards:
            _, mean, (lwr, upr) = self.number_of_state_from_ward([PatientState.REJECTED, PatientState.IN_WRONG_WARD], [ward])
            means.append(mean)
            lwrs.append(lwr)
            uprs.append(upr)
        index = [ward.name for ward in wards]
        return pd.DataFrame({'mean': means, 'lwr': lwrs, 'upr': uprs}, index=index).copy()
    
    def expected_penalty(self, wards: 'list[WardType]' = list(WardType)):
        means, lwrs, uprs = [], [], []
        for ward in wards:
            _, mean, (lwr, upr) = self.penalty_from_wards(ward)
            means.append(mean)
            lwrs.append(lwr)
            uprs.append(upr)
        index = [ward.name for ward in wards]
        return pd.DataFrame({'mean': means, 'lwr': lwrs, 'upr': uprs}, index=index).copy()


class HospitalSimulation:

    def __init__(self, config=HospitalConfiguration):
        self.ward_configs = config.wards_config
        self.switch_config = config.switch_config
        self.wardlist = config.ward_list
        self.wards = [Ward(ward, self.ward_configs[ward]) for ward in config.ward_list]
        self.patients = []
        self.arr_dist = config.inter_arrival_time_distribution
        self.stay_dist = config.stay_time_distribution
        self.total_penalty = 0
        self.time = 0
        self.patient_q: 'list[Patient]' = []

    @property
    def wardlist_without_F(self):
        return [ward for ward in self.wardlist if ward is not WardType.F]
    
    def reset_sim(self):
        self.wards = {ward: Ward(ward, self.ward_configs[ward]) for ward in self.wardlist}
        self.patients = []
        self.total_penalty = 0
        self.time = 0
        self.patient_q = []
    
    def update_wards(self):
        """Checks all wards if they have patients who are cured and removes them from the patients list"""
        for ward in self.wards.values():
            ward.update_patients(self.time)

    def switch_ward(self, patient: Patient):
        """Assigns a patient to a non-preferred ward if possible, else the patient is rejected"""
        self.total_penalty += self.ward_configs.ward(patient.preferred_ward).urgency
        new_ward = self.choose_new_ward(patient)
        new_ward.process_patient(patient)

    def choose_new_ward(self, patient: Patient):
        "Returns a ward at random according to the switch probabilities"
        ward_type = np.random.choice(a = list(WardType), p = self.switch_config[patient.preferred_ward])
        return self.wards[ward_type]

    def sim_arr(self, mean_arrival_time):
        """Returns an arrival time at random according to the inter arrival time distribution"""
        return self.time + self.arr_dist(mean_arrival_time)

    def sim_stay(self, mean_stay_time):
        """Returns a stay time at random according to the stay time distribution"""
        return self.stay_dist(mean_stay_time)
      
    def simulate_expected_penalty(self, n, expected_penalty, stoptime=365):
        allocate = lambda: self.allocate_bed_to_f_expected_penalty(expected_penalty)
        return self.sim_multiple_with_f(
            n,
            allocate,
            self.relocate_none,
            stoptime=365
        )
    
    def simulate_already_stolen(self, n, stoptime=365):
        return self.sim_multiple_with_f(
            n,
            self.allocate_bed_to_f_missing,
            self.relocate_none,
            stoptime=365
        )

    def simulate_occupation_and_penalty(self, n, stoptime=365):
        return self.sim_multiple_with_f(
            n,
            self.allocate_bed_to_f_occ_penalty,
            self.relocate_none,
            stoptime=365
        )

    def simulate_only_occupation(self, n, stoptime=365):
        return self.sim_multiple_with_f(
            n, 
            self.allocate_bed_to_f_occupation, 
            self.relocate_none,
            stoptime=365)

    
    def sim_multiple_with_f(self, n, allocation, relocation, stoptime=365):
        result = []
        for i in range(n):
            print(f'{i+1}/{n}', end='\r')
            result.append(self.simulate_with_f( allocation, relocation, display=False, stoptime=stoptime))
        return SimulationsSummary(result)

    def sim_multiple_without_f(self, n, stoptime=365):
        result = []
        for i in range(n):
            print(f'{i+1}/{n}', end='\r')
            res = self.simulate_without_f(display=False, stoptime=stoptime)
            result.append(res)
        return SimulationsSummary(result)
    
    def sim_multiple_burnin(self, n, burnin=365, simulation_length=365):
        result  =[]
        for i in range(n):
            print(f'{i+1}/{n}', end='r')
            res = self.simulate_with_burnin(display=False, burnin=burnin, simulation_length=simulation_length)
            result.append(res)
        return SimulationsSummary(result)
    
    def simulate_with_f(self, allocation, relocation, reset=True, display=True, stoptime=365):
        """Simulate a year in the given hospital"""
        if reset:
            self.reset_sim()
            new_patients = self.sim_patients(self.wards.keys())
            self.update_patient_q([p for p in new_patients])

        while self.time <= stoptime:
            if display:
                print(f'{self.time/365 * 100:.0f}%', end='\r')
            patient = heapq.heappop(self.patient_q)
            self.update_time(patient)
            self.simulate_new_patient_to_q(patient)
            self.update_wards()
            if patient.preferred_ward is WardType.F:
                self.assign_f_patient(patient, allocation)
            else:
                relocation
                self.assign_patient_to_ward(patient)
            
            self.patients.append(patient)
        
        return SimulationResult(self.patients.copy(), self.total_penalty, self.ward_configs)

    def simulate_without_f(self, reset=True, display=True, stoptime=365):
        if reset:
            self.reset_sim()
            new_patients = self.sim_patients(self.wardlist_without_F)
            self.update_patient_q([p for p in new_patients])

        while self.time <= stoptime:
            if display:
                print(f'{self.time/365 * 100:.0f}%', end='\r')
            patient = heapq.heappop(self.patient_q)
            self.update_time(patient)
            self.simulate_new_patient_to_q(patient)
            self.update_wards()
            self.assign_patient_to_ward(patient)
            self.patients.append(patient)
        
        return SimulationResult(self.patients.copy(), self.total_penalty, self.ward_configs)

    def simulate_with_burnin(self, allocation, relocation, burnin=365, simulation_length = 365, display=True):
        self.simulate_without_f(stoptime=365, dsiplay=display)
        self.total_penalty = 0
        self.patients = []
        self.simulate_with_f(allocation=allocation, relocation=relocation, reset=False, display=display, stoptime=burnin + simulation_length)
        return SimulationResult(self.patients.copy(), self.total_penalty, self.ward_configs)

    def simulate_occupation_steal(self, stoptime = 365, display=True, reset = True):
        if reset:
            self.reset_sim()
            new_patients = self.sim_patients(self.wards.keys())
            self.update_patient_q([p for p in new_patients])

        while self.time <= stoptime:
            if display:
                print(f'{self.time/365 * 100:.0f}%', end='\r')
            patient = heapq.heappop(self.patient_q)
            self.update_time(patient)
            self.simulate_new_patient_to_q(patient)
            self.update_wards()
            if patient.preferred_ward is WardType.F:
                self.assign_f_patient(patient, self.allocate_bed_to_f_occupation)
            else:
                self.assign_patient_to_ward(patient)
            
            self.patients.append(patient)
        
        return SimulationResult(self.patients.copy(), self.total_penalty, self.ward_configs)
    
    def update_time(self, patient: Patient):
        self.time = patient.arrival_time
    
    def simulate_new_patient_to_q(self, patient: Patient):
        new_patient = self.sim_single_patient(patient.preferred_ward)
        self.update_patient_q(new_patient)

    def sim_patients(self, wards):
        """Simulates patients of a certain type"""
        return [self.sim_single_patient(ward) for ward in wards]

    def sim_single_patient(self, ward):
        arr, stay = self.get_arr_and_stay_time(ward)
        return Patient(preferred_ward=ward, arrival_time=arr, stay_time=stay)

    def get_arr_and_stay_time(self, ward):
        conf = self.ward_configs.ward(ward)
        return self.sim_arr(conf.mean_arrival_time), self.sim_stay(conf.mean_stay_time)

    def update_patient_q(self, new_patients: Union['list[Patient]', Patient]):
        """Adds new patients to the correct spot in the queue"""
        if isinstance(new_patients, list):
            for patient in new_patients:
                heapq.heappush(self.patient_q, patient)
        else:
            heapq.heappush(self.patient_q, new_patients)


    def assign_patient_to_ward(self, patient: Patient) -> None:
        """If possible, assigns a patient to their preferred ward, else rellocates them"""
        ward = self.wards[patient.preferred_ward]
        if ward.process_patient(patient):
            self.switch_ward(patient)


    def assign_f_patient(self, patient: Patient, allocation_algorithm):
        ward = self.wards[WardType.F.value]
        if ward.accepted_fraction(next_is_rejected=1) <= 0.95:
            allocation_algorithm()
        self.assign_patient_to_ward(patient)
    

    def allocate_bed_to_f_occ_penalty(self):
        self.update_wards()
        f_ward = self.wards.get(WardType.F)
        other_wards = [ward for type, ward in self.wards.items() if type is not WardType.F]
        wards = sorted(other_wards, key=lambda ward: (ward.rellocation_score, ward.urgency))
        for ward in wards:
            if not ward.is_full and ward.capacity > 0:
                ward.capacity -= 1
                f_ward.capacity += 1
                return

    def allocate_bed_to_f_occupation(self):
        self.update_wards()
        f_ward = self.wards.get(WardType.F)
        other_wards = [ward for type, ward in self.wards.items() if type is not WardType.F]
        wards = sorted(other_wards, key=lambda ward: ward.fill_fraction)
        for ward in wards:
            if not ward.is_full and ward.capacity > 0:
                ward.capacity -= 1
                f_ward.capacity += 1
                return
    
    def allocate_bed_to_f_missing(self):
        self.update_wards()
        f_ward = self.wards.get(WardType.F)
        other_wards = [ward for type, ward in self.wards.items() if type is not WardType.F]
        wards = sorted(other_wards, key=lambda ward: self.ward_configs[ward.type].bed_capacity - ward.capacity)
        for ward in wards:
            if not ward.is_full and ward.capacity > 0:
                ward.capacity -= 1
                f_ward.capacity += 1
                return
    
    def allocate_bed_to_f_expected_penalty(self, expected_penalty):
        self.update_wards()
        f_ward = self.wards.get(WardType.F)
        other_wards = [ward for type, ward in self.wards.items() if type is not WardType.F]
        wards = sorted(other_wards, key=lambda ward: expected_penalty[ward.type])
        for ward in wards:
            if not ward.is_full and ward.capacity > 0:
                ward.capacity -= 1
                f_ward.capacity += 1
                return
    
    def relocate_bed_from_f(self):
        self.update_wards()
        f_ward = self.wards.get(WardType.F)
        if f_ward.capacity == 0 or f_ward.is_full:
            return

        other_wards = [ward for type, ward in self.wards.items() if type is not WardType.F]
        max_ward = max(other_wards, key=lambda ward: (ward.rellocation_score, ward.urgency))
        
        f_ward.capacity -= 1
        max_ward.capacity += 1
    
    def relocate_none(self):
        return
        
    def rellocate_bed_from_F_greedy(self):
        self.update_wards()
        f_ward = self.wards.get(WardType.F)
        if f_ward.capacity == 0 or f_ward.is_full:
            return

        other_wards = [ward for type, ward in self.wards.items() if type is not WardType.F]
        max_ward = max(other_wards, key=lambda ward: (ward.fill_fraction, ward.urgency))
        
        f_ward.capacity -= 1
        max_ward.capacity += 1
    
    def rellocate_inverse_meassure(self):
        pass


def hist_performance(sim: SimulationsSummary):
    admission = sim.expected_admissions()['mean']
    rejection = sim.expected_relocations()['mean']
    penalty = sim.expected_penalty()['mean']
    df = pd.DataFrame({
        'Expected Admissions': admission,
        'Expected Relocations': rejection,
        'Expected Urgency': penalty,
    })
    sns.barplot(data=df.T)


def hist_comp_plot(data1, data2, legend1='', legend2=''):
    df = pd.DataFrame({
    legend1: data1,
    legend2: data2
    })
    sns.histplot(df)
    return

def hist_plot(data, legend=''):
    df = pd.DataFrame({
        legend: data
    })
    sns.histplot(df)
    return


def barplot(df=None, wards = list(WardType), label=''):
    g = sns.barplot(x=[ward.name for ward in wards], y=df['mean'])
    g.set_ylabel(label)
    g.set_xlabel('Wards')
    g.set_title(label)