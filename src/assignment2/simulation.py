from dataclasses import dataclass, field
from enum import Enum, auto
import heapq




class Ward(Enum):
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()


class PatientType(Enum):
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()

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
    
    def simulate_year(self, bed_distribution=None):
        if bed_distribution is not None:
            self.bed_dist = bed_distribution

        arrival_times = heapq.heapify([0])
        regular, intensive, other = self.sim_patients(type = 'all')
        while arrival_times[0][0] <= 365:
            pass

    
        