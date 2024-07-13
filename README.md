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

# Execution
Users should execute commands from the root directory with Python 3 to start training/evaluation processes.

Following is an example of training the IQL algorithm on the 3t scenario with a total running step of 2,050,000:

```sh
python3 src/main.py --config=iql --env-config=shcfc_beta with env_args.map_name=3t t_max=2050000
```

Following is an example of training MADDPG on the 4t_vs_4t task with complex terrain features and a total running step of 10,050,000:
```sh
python3 src/main.py --config=maddpg --env-config=lhcfcws with env_args.map_name=4t_vs_4t_large_complex env_args.final_target_index=13 env_args.obs_pathing_grid='True' cuda_id='cuda:0' t_max=10050000
```

In the above examples, the config option specifies the configuration of the selected MARL algorithm, the env-config option specifies the environment, and the **map_name** states the task(map) for training.
For the MOSMAC scenarios with single-task settings, users should select **shcfc_beta** as the env-config option.
For the MOSMAC scenarios with sequential task allocation, users should select **lhcfcws** as the env-config option.

# Cite MOSMAC
If you use MOSMAC in your research, please cite our AAMAS'24 paper: [Benchmarking MARL on Long Horizon Sequential Multi-Objective Tasks](https://dl.acm.org/doi/10.5555/3635637.3663133).

*[Minghong Geng](https://gengminghong.github.io/), [Shubham Pateria](https://spateria.github.io/), Budhitama Subagdja, and [Ah-Hwee Tan](https://sites.google.com/smu.edu.sg/ahtan). 2024.
Benchmarking MARL on Long Horizon Sequential Multi-Objective Tasks.
In Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems (AAMAS '24).
International Foundation for Autonomous Agents and Multiagent Systems, Richland, SC, 2279–2281.*

In BibTeX format:

```tex
@inproceedings{10.5555/3635637.3663133,
    author = {Geng, Minghong and Pateria, Shubham and Subagdja, Budhitama and Tan, Ah-Hwee},
    title = {Benchmarking MARL on Long Horizon Sequential Multi-Objective Tasks},
    year = {2024},
    isbn = {9798400704864},
    publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
    address = {Richland, SC},
    booktitle = {Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems},
    pages = {2279–2281},
    numpages = {3},
    location = {<conf-loc>, <city>Auckland</city>, <country>New Zealand</country>, </conf-loc>},
    series = {AAMAS '24}
}
```

MOSMAC is implemented the  [Extended PyMARL (EPyMARL) framework](https://github.com/uoe-agents/epymarl).
If you use EPyMARL in your research, please cite the following paper: [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/abs/2006.07869)

*Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, & Stefano V. Albrecht. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021*

In BibTeX format:

```tex
@inproceedings{papoudakis2021benchmarking,
   title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
   author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
   booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
   year={2021},
   url = {http://arxiv.org/abs/2006.07869},
   openreview = {https://openreview.net/forum?id=cIrPX-Sn5n},
   code = {https://github.com/uoe-agents/epymarl},
}
```
