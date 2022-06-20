from dataclasses import dataclass
from enum import Enum, auto
import heapq


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

@dataclass
class Patient:
    type: PatientType
    ward: Ward
    arrival_time: float
    stay_time: dict


class HospitalSimulation:

    def __init__(self, arrival_time_dist, stay_time_dist, bed_distribution):
        self.regular_patients = []
        self.intensive_patients = []
        self.other_patients = []
        self.arr_dist = arrival_time_dist
        self.stay_dist = stay_time_dist
        self.bed_dist = bed_distribution

    def ward_switch(patientType, P=P):
        return np.random.choice(P[patientType][:])
    
    def simulate_year(self, bed_distribution=None):
        if bed_distribution is not None:
            self.bed_dist = bed_distribution

        arrival_times = heapq.heapify([0])
        regular, intensive, other = self.sim_patients()
        self.update_arrival_times([regular, intensive, other])
        while arrival_times[0] <= 365:
            



        



