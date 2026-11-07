from typing import Dict
import numpy as np


class FixedTimeController:
    def __init__(self, cycle_interval: int = 30, num_phases: int = 4):
        self.cycle_interval = cycle_interval
        self.num_phases = num_phases
        self.current_phase = 0

    def select_action(self, time_step: int, elapsed_time: int) -> int:
        if elapsed_time >= self.cycle_interval:
            self.current_phase = (self.current_phase + 1) % self.num_phases
        return self.current_phase


class RandomController:
    def __init__(self, num_phases: int = 4):
        self.num_phases = num_phases

    def select_action(self, valid_mask: np.ndarray) -> int:
        return int(np.random.choice(np.where(valid_mask == 1)[0]))


class MaxPressureController:
    def __init__(self, num_phases: int = 4):
        self.num_phases = num_phases

    def select_action(
        self, incoming_queues: Dict[str, int], downstream_queues: Dict[str, int], valid_mask: np.ndarray
    ) -> int:
        p = np.zeros(self.num_phases)
        in_q = list(incoming_queues.values())
        out_q = list(downstream_queues.values())

        p[0] = (in_q[0] - out_q[0]) + (in_q[2] - out_q[2])
        p[1] = (in_q[0] - out_q[1]) + (in_q[2] - out_q[3])
        p[2] = (in_q[1] - out_q[1]) + (in_q[3] - out_q[3])
        p[3] = (in_q[1] - out_q[0]) + (in_q[3] - out_q[2])

        p[valid_mask == 0] = -1e9
        return int(np.argmax(p))
