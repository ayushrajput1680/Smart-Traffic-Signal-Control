from typing import Dict, List, Tuple
import numpy as np


class TrafficSignalSafetyWrapper:
    def __init__(self, min_green: int = 10, yellow_time: int = 3):
        self.min_green = min_green
        self.yellow_time = yellow_time

    def get_valid_action_mask(self, current_phase: int, elapsed_time: int, num_phases: int) -> np.ndarray:
        mask = np.zeros(num_phases, dtype=np.int32)
        if elapsed_time < self.min_green:
            mask[current_phase] = 1
        else:
            mask[:] = 1
        return mask


class SimulatedIntersection:
    def __init__(self, node_id: str, incoming_lanes: List[str], outgoing_lanes: List[str]):
        self.node_id = node_id
        self.incoming_lanes = incoming_lanes
        self.outgoing_lanes = outgoing_lanes
        self.num_phases = 4
        self.current_phase = 0
        self.elapsed_time = 0
        self.queues = {lane: 0 for lane in incoming_lanes}
        self.delays = {lane: 0.0 for lane in incoming_lanes}
        self.velocities = {lane: 13.89 for lane in incoming_lanes}
        self.downstream_queues = {lane: 0 for lane in outgoing_lanes}


class MultiIntersectionEnv:
    """Lightweight queue-based multi-intersection simulator (not SUMO)."""

    def __init__(
        self,
        num_intersections: int = 4,
        decision_interval: int = 5,
        min_green: int = 10,
        yellow_time: int = 3,
        episode_length: int = 3600,
    ):
        self.num_intersections = num_intersections
        self.decision_interval = decision_interval
        self.episode_length = episode_length
        self.current_sim_step = 0
        self.safety_wrapper = TrafficSignalSafetyWrapper(min_green, yellow_time)

        self.agent_ids = [f"intersection_{i}" for i in range(num_intersections)]
        self.intersections: Dict[str, SimulatedIntersection] = {}

        for aid in self.agent_ids:
            in_lanes = [f"{aid}_in_{k}" for k in range(4)]
            out_lanes = [f"{aid}_out_{k}" for k in range(4)]
            self.intersections[aid] = SimulatedIntersection(aid, in_lanes, out_lanes)

        self.state_dim = 4 * 2 + 4 + 1 + 4
        self.action_dim = 4

    def reset(self, seed: int = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            np.random.seed(seed)
        self.current_sim_step = 0

        states = {}
        for aid, inter in self.intersections.items():
            inter.current_phase = 0
            inter.elapsed_time = 0
            for l in inter.incoming_lanes:
                inter.queues[l] = np.random.randint(0, 5)
                inter.delays[l] = 0.0
                inter.velocities[l] = 13.89
            for l in inter.outgoing_lanes:
                inter.downstream_queues[l] = np.random.randint(0, 3)
            states[aid] = self._get_agent_observation(aid)

        return states

    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        inter = self.intersections[agent_id]
        q_vals = [inter.queues[l] for l in inter.incoming_lanes]
        v_vals = [inter.velocities[l] for l in inter.incoming_lanes]
        p_one_hot = [1 if inter.current_phase == p else 0 for p in range(inter.num_phases)]
        dwell = [inter.elapsed_time / 60.0]
        neighbor_q = [np.mean(list(inter.downstream_queues.values())) for _ in range(4)]
        return np.array(q_vals + v_vals + p_one_hot + dwell + neighbor_q, dtype=np.float32)

    def get_action_masks(self) -> Dict[str, np.ndarray]:
        return {
            aid: self.safety_wrapper.get_valid_action_mask(inter.current_phase, inter.elapsed_time, inter.num_phases)
            for aid, inter in self.intersections.items()
        }

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict]:
        self.current_sim_step += self.decision_interval
        rewards = {}

        for aid, action in actions.items():
            inter = self.intersections[aid]
            if action == inter.current_phase:
                inter.elapsed_time += self.decision_interval
            else:
                inter.current_phase = action
                inter.elapsed_time = self.decision_interval

            phase = inter.current_phase
            for idx, lane in enumerate(inter.incoming_lanes):
                new_arrivals = np.random.poisson(0.8)
                discharged = 0
                if (phase == 0 and idx in [0, 2]) or (phase == 2 and idx in [1, 3]):
                    discharged = min(inter.queues[lane], np.random.randint(2, 4))
                inter.queues[lane] = max(0, inter.queues[lane] + new_arrivals - discharged)
                inter.delays[lane] = inter.queues[lane] * 1.5
                inter.velocities[lane] = 13.89 if inter.queues[lane] == 0 else 2.5

            local_queue_sum = sum(inter.queues.values())
            local_delay_sum = sum(inter.delays.values())
            downstream_sum = sum(inter.downstream_queues.values())
            rewards[aid] = float(-(local_queue_sum + 0.1 * local_delay_sum) - 0.2 * downstream_sum)

        next_states = {aid: self._get_agent_observation(aid) for aid in self.agent_ids}
        done_flag = self.current_sim_step >= self.episode_length
        dones = {aid: done_flag for aid in self.agent_ids}

        avg_wait = np.mean([sum(i.delays.values()) for i in self.intersections.values()])
        avg_q = np.mean([sum(i.queues.values()) for i in self.intersections.values()])
        info = {"avg_waiting_time": avg_wait, "avg_queue_length": avg_q}

        return next_states, rewards, dones, info
