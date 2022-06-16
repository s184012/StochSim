from dataclasses import dataclass
from enum import Enum, auto
from math import factorial
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns


class State(Enum):
    INCOMING = auto()
    IN_SERVICE = auto()
    SERVICED = auto()
    BLOCKED = auto()

@dataclass
class Event:
    state: State
    arrival_time: int
    departure_time: int


class EventList:

    events: 'list[Event]'
    in_service: 'list[Event]'

    def __init__(self, arrival_time_distribution: stats.rv_continuous, service_time_distribution: stats.rv_continuous, number_of_events):
        self._arr_dist = arrival_time_distribution
        self._serv_dist = service_time_distribution
        self.events = self.generate_events(number_of_events)
        self.time_line = {}
        self.in_service = []
    
    @property
    def states(self):
        return [event.state for event in self.events]
    
    def update_in_service(self, time):
        for event in self.in_service:
            if event.departure_time <= time:
                event.state = State.SERVICED
        
        self.in_service = [event for event in self.events if event.state == State.IN_SERVICE]

    def generate_events(self, number_of_events: int) -> 'list[Event]':
        arr_times = self._arr_dist.rvs(size=number_of_events)
        arr_times = np.cumsum(arr_times)

        serv_times = self._serv_dist.rvs(size=number_of_events)
        dep_times = arr_times + serv_times
        return [Event(State.INCOMING, arr, dep) for arr, dep in zip(arr_times, dep_times)]

    def update_timeline(self, time):
        self.time_line[time] = self.states
    
    def __str__(self):
        str = ''
        for iter, (time, states) in enumerate(self.time_line.items()):
            if iter < len(self.events) - 10:
                continue
            str += f'TIME: {time:.2f}\n'
            for i, state in enumerate(states):
                if state != State.INCOMING:
                    str += f'Obs. {i}: {state.name}\n'
            str += '\n'
        return str

class BlockingEventSimulation:

    def __init__(self, arrival_time_distribution: stats.rv_continuous, service_time_distribution: stats.rv_continuous):
        self.arrival_dist = arrival_time_distribution
        self.service_dist = service_time_distribution

    def simulate(self, max_events: int, service_units: int):
        blocked_count = 0
        event_list = EventList(self.arrival_dist, self.service_dist, max_events)
        for event in event_list.events:
            time = event.arrival_time
            event_list.update_in_service(time)
            if len(event_list.in_service) < service_units:
                event.state = State.IN_SERVICE
                event_list.update_in_service(time)
            else:
                event.state = State.BLOCKED
                blocked_count += 1

    
            event_list.update_timeline(time)
        
        return blocked_count / max_events


def calculate_theoretical_block_pct(m, a):
    return (a**10/factorial(m))/ sum([a**i / factorial(int(i)) for i in range(m+1)])

if __name__ == '__main__':
    arr = stats.expon()
    serv = stats.expon(scale=8)
    sim = BlockingEventSimulation(arr, serv)
    event, block_pct = sim.simulate(10_000, 10)
    a=8
    print((a**10/factorial(10))/ sum([a**i / factorial(int(i)) for i in range(10+1)]), block_pct)

   