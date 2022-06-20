from dataclasses import dataclass, field
from enum import Enum, auto
import heapq
import numpy as np

P = np.zeros([6,6])
P[0, :] = [0.0, 0.05, 0.10, 0.05, 0.80, 0.0]
P[1, :] = [0.20, 0.0, 0.50, 0.15, 0.15, 0.0]
P[2, :] = [0.30, 0.20, 0.0, 0.20, 0.30, 0.0]
P[3, :] = [0.35, 0.30, 0.05, 0.0, 0.30, 0.0]
P[4, :] = [0.20, 0.10, 0.60, 0.10, 0.0, 0.0] 
P[5, :] = [0.20, 0.20, 0.20, 0.20, 0.20, 0.0]

arr_Times = [14.5, 11.0, 8.0, 6.5, 5.0, 13.0]

len_stay = [2.9, 4.0, 4.5, 1.4, 3.9, 2.2]

urgency = [7, 5 , 2, 10, 5]

class Ward(Enum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5


class PatientType(Enum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5

@dataclass
class BedDistribution:
    A: int
    B: int
    C: int
    D: int
    E: int
    F: int


@dataclass(order=True)
class Patient:
    type: PatientType=field(compare = False)
    ward: Ward=field(compare = False)
    arrival_time: float
    stay_time: dict=field(compare = False)


class HospitalSimulation:

    def __init__(self, arrival_time_dist, stay_time_dist, bed_distribution):
        self.regular_patients = []
        self.intensive_patients = []
        self.other_patients = []
        self.arr_dist = arrival_time_dist
        self.stay_dist = stay_time_dist
        self.bed_dist = bed_distribution

    def ward_switch(type, P=P):
        return np.random.choice(a = np.range(6), p = P[type][:])
    
    def simulate_year(self, bed_distribution=None):
        if bed_distribution is not None:
            self.bed_dist = bed_distribution

        arrival_times = []
        regular, intensive, other = self.sim_patients(type='all')
        self.update_arrival_times([regular, intensive, other])
        while arrival_times[0] <= 365:
            pass
        
    
    def sim_patients(type = 'all'):
        if (type == 'all'):

        else:

        patients = Patient 
        return patients


    
        