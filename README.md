# MOSMAC: A Multi-agent Reinforcement Learning Benchmark on Sequential Multi-Objective Tasks

# Features
MOSMAC provides a set of multi-objective multi-agent reinforcement learning (MOMARL) tasks 
extending from [SMAC](https://github.com/oxwhirl/smac), which originally focused on 
single-objective multi-agent reinforcement learning (MARL) tasks in the StarCraft II environment.

Specifically, MOSMAC includes the following features:
- **Multi-Objective Tasks**: Reinforcement learning agents in MOSMAC scenarios need to learn
policies that simultaneously balance multiple objectives beyond combat. Specifically, MOSMAC
considers objectives including:
  - Combat: a widely adopted objective in SC2 environment, originally presented by Samvelyan 
et al. in [SMAC (2019)](https://github.com/oxwhirl/smac),
  - Safety: also know as _escape_  by Hu et al. in [MO-MIX (2023)](https://ieeexplore.ieee.org/document/10145811/),
  - Navigate: a novel objective where agents need to navigate to a target location, 
firstly presented in our [AAMAS paper (2024)](https://dl-acm-org.libproxy.smu.edu.sg/doi/10.5555/3635637.3663133).   
- **Sequential Task Allocation**: As a closer simulation to real-world scenarios, MOSMAC 
includes a set of scenarios that challenge agents with sequential task allocation, where 
agents need to complete multiple tasks sequentially.
