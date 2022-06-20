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
class Patient:
    type: PatientType
    ward: Ward
    stay_time: dict


class HospitalSimulation:
    pass