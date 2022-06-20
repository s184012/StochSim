from dataclasses import dataclass
from enum import Enum, auto


class Ward(Enum):
    A = auto()
    B = auto()
    C = auto()
    OTHER = auto()

class PatientType(Enum):
    REGULAR = auto()
    INTENSIVE = auto()
    OTHER = auto()

@dataclass
class BedDistribution:
    A: int
    B: int
    C: int

@dataclass
class Patient:
    type: PatientType
    ward: Ward
    stay_time: dict


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
        
        


