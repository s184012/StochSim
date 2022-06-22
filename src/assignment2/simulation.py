from dataclasses import dataclass, field
from enum import Enum, auto
import heapq
from typing import Union
import numpy as np
from scipy import stats

P = np.zeros([6,6])
P[0, :] = [0.0, 0.05, 0.10, 0.05, 0.80, 0.0]
P[1, :] = [0.20, 0.0, 0.50, 0.15, 0.15, 0.0]
P[2, :] = [0.30, 0.20, 0.0, 0.20, 0.30, 0.0]
P[3, :] = [0.35, 0.30, 0.05, 0.0, 0.30, 0.0]
P[4, :] = [0.20, 0.10, 0.60, 0.10, 0.0, 0.0] 
P[5, :] = [0.20, 0.20, 0.20, 0.20, 0.20, 0.0]

arr_Times = [14.5, 11.0, 8.0, 6.5, 5.0, 13.0]

len_stay = [2.9, 4.0, 4.5, 1.4, 3.9, 2.2]

urgency = [7, 5, 2, 10, 5, 0]

bed_capacity = [55, 40, 30, 20, 20, 0]

class SimulationConfig:

    def __init__(self, switch_probabilities, mean_arrival_times, mean_stay_time, urgency, bed_capacities):
        pass



class WardType(Enum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5


class PatientState(Enum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5
    CURED = 6
    REJECTED = 7


@dataclass(order=True)
class Patient:
    state: PatientState=field(compare = False)
    ward: WardType=field(compare = False, init=False)
    penalty: int=field(compare=False, init=False)
    arrival_time: float
    stay_time: dict=field(compare = False)

    def __post_init__(self):
        self.penalty = urgency[self.state.value]
        self.ward = WardType(self.state.value)



@dataclass
class Ward:
    ward_type: WardType
    capacity: int = field(init=False)
    urgency: int = field(init=False)
    total_number_of_treated_patients: int = 0
    rejected_patients: int = 0
    patients: list = field(default_factory=list)
    
    def __post_init__(self):
        self.capacity = bed_capacity[self.ward_type.value]
        self.urgency = urgency[self.ward_type.value]

    @property
    def number_of_current_patients(self):
        return len(self.patients)

    @property
    def is_full(self):
        return self.number_of_current_patients == self.capacity
    
    @property
    def rejected_fraction(self):
        return self.rejected_patients / self.total_number_of_treated_patients

    def add_patient(self, patient: Patient):
        heapq.heappush(self.patients, (patient.arrival_time + patient.stay_time, patient))
        self.total_number_of_treated_patients += 1

    def reject_patient(self, patient: Patient):
        patient.state = PatientState.REJECTED
        self.reject_patient += 1

    
    def update_patients(self, curTime):
        if not self.patients:
            return

        while len(self.patients) != 0 and self.patients[0][0] <= curTime:
            self.patients[0][1].type = PatientState.CURED
            heapq.heappop(self.patients)


@dataclass
class BedDistribution:
    A: int
    B: int
    C: int
    D: int
    E: int
    F: int





class HospitalSimulation:

    def __init__(self, arrival_time_dist, stay_time_dist, bed_distribution=bed_capacity):
        self.wards = {ward: Ward(ward) for ward in WardType if ward is not WardType.F}
        self.arr_dist = arrival_time_dist
        self.stay_dist = stay_time_dist
        self.bed_dist = bed_distribution
        self.total_penalty = 0

    def switch_ward(self, patient: Patient, P=P):
        self.total_penalty += patient.penalty
        next_ward = self.choose_new_ward(patient, P)
        if next_ward.is_full:
            next_ward.reject_patient(patient)
        else:
            next_ward.add_patient(patient)

    def choose_new_ward(self, patient: Patient, P=P):
        ward_type = np.random.choice(a = list(WardType), p = P[patient.state.value,:])
        return self.wards[ward_type]

    def sim_arr(self,curTime, arr_Time):
        return curTime + self.arr_dist.rvs(scale=1/arr_Time)

    def sim_stay(self,stay):
        return self.stay_dist.rvs(scale = stay)

    def simulate_year(self, pType = 'nof', bed_distribution=None):
        if bed_distribution is not None:
            self.bed_dist = bed_distribution
        self.total_penalty = 0
        patient_q = self.sim_patients(type= pType, curTime=0)
        heapq.heapify(patient_q)
        t = 0
        while t <= 365:
            patient = heapq.heappop(patient_q)
            t = patient.arrival_time
            new_patient = self.sim_patients(type=patient.type, curTime=t)

            self.assign_patient_to_ward(patient, curTime=t)

            self.update_patient_q(patient_q, new_patient)


    def sim_patients(self, curTime, type = 'all'):
        if (type == 'all' or type == 'nof'):
            patients = []
            for pType, arr_time, stay_time in zip(list(PatientState), arr_Times, len_stay):
                if (pType is PatientState.F and type == 'nof'):
                    break
                patient = Patient(state=pType, arrival_time = self.sim_arr(curTime, arr_Time = arr_time), stay_time = self.sim_stay(stay=stay_time))
                patients.append(patient)
        else:
            patients = Patient(state = type, arrival_time = self.sim_arr(curTime, arr_Time = arr_Times[type.value]), stay_time = self.sim_stay(stay=len_stay[type.value]))
        return patients


    def update_patient_q(self, heap, new_patients: Union['list[Patient]', Patient]):
        if isinstance(new_patients, list):
            for patient in new_patients:
                heapq.heappush(heap, patient)
        else:
            heapq.heappush(heap, new_patients)


    def assign_patient_to_ward(self, patient: Patient, curTime) -> None:
        ward = self.wards[patient.ward]
        ward.update_patients(curTime = curTime)
        if ward.is_full:
            self.switch_ward(patient)
        else:
            ward.add_patient(patient)