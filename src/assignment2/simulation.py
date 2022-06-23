from dataclasses import dataclass, field
from enum import IntEnum, Enum, auto
import heapq
from typing import Callable, TypedDict, Union
import numpy as np
from scipy import stats

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
    IN_CORRECT_YARD = 1
    IN_WRONG_YARD = 2
    CURED = 3
    REJECTED = 4


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
    arrival_time: float
    stay_time: dict=field(compare = False)
    state: PatientState=field(compare = False, default=PatientState.PENDING)
    assigned_ward: WardType = None

    def update_ward(self, ward: WardType):
        self.assigned_ward = ward
        if self.assigned_ward is self.preferred_ward:
            self.state = PatientState.IN_CORRECT_YARD
        else:
            self.state = PatientState.IN_WRONG_YARD


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
            self.patients[0][1].type = PatientState.CURED
            heapq.heappop(self.patients)



class SimulationResult:
    """Contains the result of a simulation"""
    def __init__(self, patients: 'list[Patient]', penalty, ward_config: WardsConfigurations) -> None:
        self.patients = patients
        self.penalty = penalty
        self.config = ward_config
        self.number_of_patients = len(patients)
    
    def state(self, state: PatientState):
        return [p for p in self.patients]

    def state_pct_total(self, state: PatientState):
        count = len([p for p in self.patients if p.state is state])
        return count / self.number_of_patients * 100

    def ward(self, ward: WardType):
        return [p for p in self.patients if p.assigned_ward is ward]
    
    def state_from_ward(self, state: PatientState, ward: WardType):
        return [p for p in self.ward(ward) if p.state is state]
    
    def state_pct_ward(self, state: PatientState, ward: WardType):
        return len(self.state_from_ward(state, ward)) / len(self.ward(ward)) * 100
    
    def relocated_from(self, ward):
        [p for p in self.state(PatientState.IN_WRONG_YARD) if p.preferred_ward is ward]
    
    def relocated_from_pct(self, ward):
        return len(self.relocated_from(ward)) / len(self.ward(ward))
    
    def penalty_from_ward(self, ward: WardType):
        return len(self.relocated_from(ward)) * self.config[ward].urgency
    


class SimulationsSummary:
    """Summarises and compares different simulation results"""
    def __init__(self, results: 'list[SimulationResult]'):
        self.results = results
    
    
    

        


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
      
    def simulate_year(self, reset=True, display=True, stoptime=365):
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
                self.assign_f_patient(patient)
            else:
                self.rellocate_bed_from_F()
                self.assign_patient_to_ward(patient)
            
            self.patients.append(patient)

    def simulate_year_without_f(self, reset=True, display=True, stoptime=365):
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

    def simulate_with_burnin(self, burnin=365, simulation_length = 365, display=True):
        self.simulate_year_without_f(stoptime=365, dsiplay=display)
        self.total_penalty = 0
        self.patients = []
        self.simulate_year(reset=False, display=display, stoptime=burnin + simulation_length)



    
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


    def assign_f_patient(self, patient: Patient):
        ward = self.wards[WardType.F]
        print(ward.accepted_fraction(next_is_rejected=0))
        if ward.accepted_fraction(next_is_rejected=1) <= 0.95:
            self.rellocate_bed_to_F()
        self.assign_patient_to_ward(patient)
    

    def rellocate_bed_to_F(self):
        self.update_wards()
        f_ward = self.wards.get(WardType.F)
        other_wards = [ward for type, ward in self.wards.items() if type is not WardType.F]
        wards = sorted(other_wards, key=lambda ward: (ward.rellocation_score, ward.urgency))
        for ward in wards:
            if not ward.is_full and ward.capacity > 0:
                ward.capacity -= 1
                f_ward.capacity += 1
                return

    def rellocate_bed_from_F(self):
        self.update_wards()
        f_ward = self.wards.get(WardType.F)
        if f_ward.capacity == 0 or f_ward.is_full:
            return

        other_wards = [ward for type, ward in self.wards.items() if type is not WardType.F]
        max_ward = max(other_wards, key=lambda ward: (ward.rellocation_score, ward.urgency))
        
        f_ward.capacity -= 1
        max_ward.capacity += 1
