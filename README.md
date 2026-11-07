# Multi-Agent Deep RL for Decentralized Urban Traffic-Signal Coordination

## Overview
Independent DQN and Shared-Parameter DQN agents controlling signalized intersections in a
4-intersection queue-based simulator, compared against Random, Fixed-Time and Max-Pressure controllers.

## Problem Formulation
Modeled as a Dec-POMDP over an intersection graph:
- **Agents:** one controller per intersection.
- **Observation (17-dim):** 4 queue lengths, 4 mean speeds, one-hot phase (4), phase dwell time, 4 downstream queue features.
- **Actions:** choose one of 4 phases, with action masking enforcing a 10 s minimum green.
- **Reward:** negative local queue + 0.1 x delay, minus 0.2 x downstream queue.

## Repository Structure
```
.
├── requirements.txt
├── src/
│   ├── agent.py       # DQN, replay buffer, target network, action masking
│   ├── env.py         # Multi-intersection simulator + safety mask
│   ├── baselines.py   # Random, Fixed-Time, Max-Pressure controllers
│   ├── train.py       # Training loop (independent / shared DQN)
│   └── plot_curves.py # Learning-curve plots (mean ± std over seeds)
└── results/           # CSV logs and learning_curves.png
```

## Usage
```bash
pip install -r requirements.txt
cd src
for seed in 1 2 3; do
  python train.py --algo shared_dqn      --seed $seed --episodes 50
  python train.py --algo independent_dqn --seed $seed --episodes 50
done
python plot_curves.py --results ../results
```

## Results
Run the commands above to produce `results/learning_curves.png`. Add a benchmark table here
only with numbers you measured from your own runs (including baselines, evaluated in the same environment).

## Limitations
The environment is a simplified queue model, not SUMO: no vehicle dynamics, yellow-phase
simulation, or inter-intersection flow (downstream queues are static after reset).
