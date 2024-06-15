from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from mosmac.env.multiagentenv import MultiAgentEnv
from mosmac.env.starcraft2.maps import get_map_params
from smac.env.starcraft2.starcraft2 import StarCraft2Env

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
# from absl import logging
import logging

# import itertools

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol
# from pysc2.lib import sc_process

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
# from s2clientprotocol import raw_pb2 as r_pb
# from s2clientprotocol import debug_pb2 as d_pb

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class StarCraft2EnvLMANC(StarCraft2Env):
    """The modified StarCraft II environment for decentralised multi-agent micromanagement scenarios.

    It inherits the methods of StarCraft2Env, which were originally introduced in the smac package."""

    def __init__(
            self,
            map_name="3t",
            step_mul=8,
            move_amount=2,
            difficulty="7",
            game_version=None,
            seed=None,
            continuing_episode=False,
            obs_all_health=True,
            obs_own_health=True,
            obs_last_action=False,
            obs_pathing_grid=False,  # [MOSMAC's new attribute] Agents' observation don't include pathing grid by
            # default. Set as True if users want to include in scenarios with complex terrain.
            obs_terrain_height=False,  # [MOSMAC's new attribute] Agents' observation don't include terrain height by
            # default. Set as True if users want to include in scenarios with complex terrain.
            obs_instead_of_state=False,
            obs_timestep_number=False,
            state_last_action=True,
            state_timestep_number=False,
            reward_sparse=False,
            reward_only_positive=True,
            reward_death_value=10,
            reward_win=200,
            reward_defeat=0,
            reward_negative_scale=0.5,
            reward_scale=True,
            reward_scale_rate=20,
            replay_dir="",
            replay_prefix="",
            window_size_x=1920,
            window_size_y=1200,
            heuristic_ai=False,
            heuristic_rest=False,
            debug=False,  # default is False. Set as True if users want to record log file.
            alpha=0.5,  # [MOSMAC's new attribute] The weights for objective 1 in dual-objective liner utility functions
            obs_direction_command=True,  # [MOSMAC's new attribute] Agents' obs include signals of target positions.

            # Parameters for hierarchical control, utilized in our paper
            # "HiSOMA: A Hierarchical Multi-Agent Model Integrating Self-Organizing Neural Networks with Multi-Agent
            # Deep Reinforcement Learning" published in Expert Systems with Applications Journal.
            # All the following attribute are new features in MOSMAC.
            hierarchical=False,  # New feature in LMANC
            n_ally_platoons=1,
            n_ally_agent_in_platoon=3,
            n_enemy_platoons=1,
            n_enemy_unit_in_platoon=3,
            map_sps=None,
            obs_distance_target=True,
            only_need_1p_arrive=None,
            reward_StrategicPoint_val=20,  # Optional rewards for moving to strategic points.
            reward_StrategicPoint_loc=[110.38, 104.69],
            final_target_index=13,  # Set final target to adjust the number of tasks in sequential task allocation.
            path_sequences=[   # optional path sequences for the agents to follow in sequential task allocation.
                [0, 1, 2, 3, 4, 6, 11, 12],
                [0, 1, 2, 3, 4, 6, 11, 12],
                [0, 1, 2, 3, 4, 6, 11, 12],
            ],
            fixed_sequence=None,
            number_of_subtask=None,  # Varying the number of subtasks
            episode_limit=None,
            last_completed_subgoal=None,
            momarl_setting=None,  # Enable tracking multi-objective MARL learning results
            reward_objectives=None
    ):
        """ Initialize a StarCraftC2EnvLMANC environment for training MARL agents to solve multi-objective multi-agent
        decision-making problems. It overwrites the __init__() method of StarCraft2Env and include more attributes for
        the MOMARL environment.

        Besides supporting the results in the AAMAS'24 paper: "Benchmarking MARL on Long Horizon Sequential
        Multi-Objective Tasks", it also includes APIs for the paper "HiSOMA: A Hierarchical Multi-Agent Model
        Integrating Self-Organizing Neural Networks with Multi-Agent Deep Reinforcement Learning" published in Expert
        Systems with Applications Journal. We will also release the codes of the HiSOMA model in the near future.

        Parameters
        ----------
        map_name : str, optional
            The name of the SC2 map to play (default is "3t"). The full list
            can be found by running bin/map_list.
        step_mul : int, optional
            How many game steps per agent step (default is 8). None
            indicates to use the default map step_mul.
        move_amount : float, optional
            How far away units are ordered to move per step (default is 2).
        difficulty : str, optional
            The difficulty of built-in computer AI bot (default is "7").
        game_version : str, optional
            StarCraft II game version (default is None). None indicates the
            latest version.
        seed : int, optional
            Random seed used during game initialisation. This allows to
        continuing_episode : bool, optional
            Whether to consider episodes continuing or finished after time
            limit is reached (default is False).
        obs_all_health : bool, optional
            Agents receive the health of all units (in the sight range) as part
            of observations (default is True).
        obs_own_health : bool, optional
            Agents receive their own health as a part of observations (default
            is False). This flag is ignored when obs_all_health == True.
        obs_last_action : bool, optional
            Agents receive the last actions of all units (in the sight range)
            as part of observations (default is False).
        obs_pathing_grid : bool, optional
            Whether observations include pathing values surrounding the agent
            (default is False).
        obs_terrain_height : bool, optional
            Whether observations include terrain height values surrounding the
            agent (default is False).
        obs_instead_of_state : bool, optional
            Use combination of all agents' observations as the global state
            (default is False).
        obs_timestep_number : bool, optional
            Whether observations include the current timestep of the episode
            (default is False).
        state_last_action : bool, optional
            Include the last actions of all agents as part of the global state
            (default is True).
        state_timestep_number : bool, optional
            Whether the state include the current timestep of the episode
            (default is False).
        reward_sparse : bool, optional
            Receive 1/-1 reward for winning/loosing an episode (default is
            False). Whe rest of reward parameters are ignored if True.
        reward_only_positive : bool, optional
            Reward is always positive (default is True).
        reward_death_value : float, optional
            The amount of reward received for killing an enemy unit (default
            is 10). This is also the negative penalty for having an allied unit
            killed if reward_only_positive == False.
        reward_win : float, optional
            The reward for winning in an episode (default is 200).
        reward_defeat : float, optional
            The reward for loosing in an episode (default is 0). This value
            should be non-positive.
        reward_negative_scale : float, optional
            Scaling factor for negative rewards (default is 0.5). This
            parameter is ignored when reward_only_positive == True.
        reward_scale : bool, optional
            Whether to scale the reward (default is True).
        reward_scale_rate : float, optional
            Reward scale rate (default is 20). When reward_scale == True, the
            reward received by the agents is divided by (max_reward /
            reward_scale_rate), where max_reward is the maximum possible
            reward per episode without considering the shield regeneration
            of Protoss units.
        replay_dir : str, optional
            The directory to save replays (default is None). If None, the
            replay will be saved in Replays directory where StarCraft II is
            installed.
        replay_prefix : str, optional
            The prefix of the replay to be saved (default is None). If None,
            the name of the map will be used.
        window_size_x : int, optional
            The length of StarCraft II window size (default is 1920).
        window_size_y: int, optional
            The height of StarCraft II window size (default is 1200).
        heuristic_ai: bool, optional
            Whether to use a non-learning heuristic AI (default False).
        heuristic_rest: bool, optional
            At any moment, restrict the actions of the heuristic AI to be
            chosen from actions available to RL agents (default is False).
            Ignored if heuristic_ai == False.
        debug: bool, optional
            Log messages about observations, state, actions and rewards for
            debugging purposes (default is False).

        Following are new input variables in StarCraft2EnvLMANC.
        hierarchical: bool, optional
            Whether to use hierarchical control architecture over multi-agent
            cooperation (default is False in non-hierarchical MARL algorithms).

        Following are the new parameters for the multi-objective MARL setting.
        momarl_setting: bool, optional
            Whether the training want to utilize a multi-objective MARL setting.
            If true, rewards for individual objectives will be tracked by self.battle() and self.step() method.
        """
        # The parameters for MOMARL setting
        self.momarl_setting = momarl_setting
        self.reward_objectives = reward_objectives.strip('[]').split(',')
        # The parameters of hierarchical control architecture over multiple agents.
        self.reward_StrategicPoint_loc = reward_StrategicPoint_loc
        self.reward_StrategicPoint_val = reward_StrategicPoint_val
        self.hierarchical = hierarchical
        self.map_sps = map_sps
        if self.map_sps is not None:
            self.matrix_size = len(map_sps) + 1
        self.target_SP_loc = None
        self.target_SP_id = None
        self.only_need_1p_arrive = only_need_1p_arrive
        self.final_target_index = final_target_index
        self.fixed_sequence = fixed_sequence
        self.last_completed_subgoal = last_completed_subgoal

        self.n_sp = 1
        self.reward_SP = 20
        self.reward_arrive = 50

        # add the distance as part of the observation
        self.obs_distance_target = obs_distance_target
        self.n_obs_distance_target = 1

        # Map arguments
        self.map_name = map_name
        map_params = get_map_params(self.map_name)
        self.n_agents = map_params["n_agents"]
        self.n_enemies = map_params["n_enemies"]

        if episode_limit:
            self.episode_limit = episode_limit
        else:
            self.episode_limit = map_params["limit"]
        self.action_limits = 50  # Top level controller sends at most 50 commands.

        # following 3 features only work in hierarchical control
        self.recent_states = []  # record recent states to check whether any platoon is stuck
        self.recent_states_size = 10  # record 8 recent states to check whether any platoon is stuck
        self.remove_stuck_platoon = True

        self._move_amount = move_amount
        self._step_mul = step_mul
        self.difficulty = difficulty

        # Observations and state
        self.obs_own_health = obs_own_health
        self.obs_all_health = obs_all_health
        self.obs_instead_of_state = obs_instead_of_state
        self.obs_last_action = obs_last_action
        self.obs_pathing_grid = obs_pathing_grid
        self.obs_terrain_height = obs_terrain_height
        self.obs_direction_command = obs_direction_command
        self.obs_timestep_number = obs_timestep_number
        self.state_last_action = state_last_action
        self.state_timestep_number = state_timestep_number
        if self.obs_all_health:
            self.obs_own_health = True
        self.n_obs_pathing = 8
        self.n_obs_height = 9
        self.n_obs_direction_command = 4  # The dimension of the direction command.

        # Rewards args
        self.reward_sparse = reward_sparse
        self.reward_only_positive = reward_only_positive
        self.reward_negative_scale = reward_negative_scale
        self.reward_death_value = reward_death_value
        self.reward_win = reward_win
        self.reward_defeat = reward_defeat
        self.reward_scale = reward_scale
        self.reward_scale_rate = reward_scale_rate

        # Other
        self.game_version = game_version
        self.continuing_episode = continuing_episode
        self._seed = seed
        self.heuristic_ai = heuristic_ai
        self.heuristic_rest = heuristic_rest
        self.debug = debug
        self.window_size = (window_size_x, window_size_y)
        self.replay_dir = replay_dir
        self.replay_prefix = replay_prefix
        self.alpha = alpha  # The weight of the first objective in the linear utility function of dual-objective MOMARL.

        # Restrict the obs of agents
        self.only_local_enemy_obs = False  # default is False
        self.only_local_enemy_state = False  # default is False
        self.n_local_enemies = 4
        self.path_sequences = path_sequences
        self.current_path = None
        self.path_id = 0
        self.scenario_id = 0

        if number_of_subtask:
            self.number_of_subtask = number_of_subtask

        # Actions
        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        if self.only_local_enemy_obs:
            self.n_actions = self.n_actions_no_attack + self.n_local_enemies
        else:
            self.n_actions = self.n_actions_no_attack + self.n_enemies

        # Map info
        self._agent_race = map_params["a_race"]
        self._bot_race = map_params["b_race"]
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = map_params["unit_type_bits"]
        self.map_type = map_params["map_type"]
        self._unit_types = None

        self.max_reward = self.n_enemies * self.reward_death_value

        # create lists containing the names of attributes returned in states
        self.ally_state_attr_names = [
            "health",
            "energy/cooldown",
            "rel_x",
            "rel_y",
        ]
        self.enemy_state_attr_names = ["health", "rel_x", "rel_y"]

        if self.shield_bits_ally > 0:
            self.ally_state_attr_names += ["shield"]
        if self.shield_bits_enemy > 0:
            self.enemy_state_attr_names += ["shield"]

        if self.unit_type_bits > 0:
            bit_attr_names = [
                "type_{}".format(bit) for bit in range(self.unit_type_bits)
            ]
            self.ally_state_attr_names += bit_attr_names
            self.enemy_state_attr_names += bit_attr_names

        self.initial_pos = []  # Record initial positions
        self.n_ally_agent_in_platoon = n_ally_agent_in_platoon
        if not self.hierarchical:
            self.n_ally_platoons = 1
            self.n_ally_agent_in_platoon = self.n_agents
        else:
            self.n_ally_platoons = n_ally_platoons
            self.n_ally_agent_in_platoon = n_ally_agent_in_platoon

        # Enemy has only 1 platoon
        self.n_enemy_platoons = n_enemy_platoons
        self.n_enemy_unit_in_platoon = n_enemy_unit_in_platoon
        if self.hierarchical:
            self.agent_reach_point = [[False] * self.n_ally_agent_in_platoon] * self.n_ally_platoons
        else:
            self.agent_reach_point = [0 for _ in range(self.n_agents)]
        self.agents_movement_record = [[False] * self.n_ally_agent_in_platoon] * self.n_ally_platoons
        self.platoons_move_record = None
        # -------------------------------------------
        self.agents = {}
        self.enemies = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self._min_unit_type = 0
        self.marine_id = self.marauder_id = self.medivac_id = 0
        self.hydralisk_id = self.zergling_id = self.baneling_id = 0
        self.stalker_id = self.colossus_id = self.zealot_id = 0
        self.max_distance_x = 0
        self.max_distance_y = 0
        self.map_x = 0
        self.map_y = 0
        self.reward = 0
        self.renderer = None
        self.terrain_height = None
        self.pathing_grid = None
        self._run_config = None
        self._sc2_proc = None
        self._controller = None

        # Try to avoid leaking SC2 processes on shutdown
        atexit.register(lambda: self.close())

    def get_distance_target(self, unit, aid=None, pid=None):
        alpha = 3

        deviation = [0, 0]  # The deviation is adjusted for each agent.
        if aid == 0:
            deviation = [0, alpha]
        if aid == 1:
            deviation = [0, -alpha]
        if aid == 2:
            deviation = [alpha, 0]
        if aid == 3:
            deviation = [-alpha, 0]
        else:
            deviation = [0, 0]

        # self.distance use math.hypot to calculate the distance
        if pid is None:
            distance = self.distance(
                unit.pos.x,
                unit.pos.y,
                (self.target_SP_loc[0] + deviation[0]),
                (self.target_SP_loc[1] + deviation[1])
            )
        else:
            distance = self.distance(
                unit.pos.x,
                unit.pos.y,
                (self.target_SP_loc[0][0] + deviation[0]),
                (self.target_SP_loc[0][1] + deviation[1])
            )

        return distance

    def get_direction_command(self, unit, aid=None, pid=None):
        vals = [0, 0, 0, 0]
        alpha = 3

        if aid == 0:
            deviation = [0, alpha]
        if aid == 1:
            deviation = [0, -alpha]
        if aid == 2:
            deviation = [alpha, 0]
        if aid == 3:
            deviation = [-alpha, 0]
        else:
            deviation = [0, 0]

        if pid is None:
            if unit.pos.x < self.target_SP_loc[0] + deviation[0]:
                vals[0] = 1  # should go east
            if unit.pos.y > self.target_SP_loc[1] + deviation[1]:
                vals[1] = 1  # should go south
            if unit.pos.x > self.target_SP_loc[0] + deviation[0]:
                vals[2] = 1  # should go west
            if unit.pos.y < self.target_SP_loc[1] + deviation[1]:
                vals[3] = 1  # should go north
        else:
            if unit.pos.x < self.target_SP_loc[pid][0] + deviation[0]:
                vals[0] = 1  # should go east
            if unit.pos.y > self.target_SP_loc[pid][1] + deviation[1]:
                vals[1] = 1  # should go south
            if unit.pos.x > self.target_SP_loc[pid][0] + deviation[0]:
                vals[2] = 1  # should go west
            if unit.pos.y < self.target_SP_loc[pid][1] + deviation[1]:
                vals[3] = 1  # should go north

        return vals

    def init_units(self):
        """Initialise the units.
        During the experiments, we notice that if the SC2 game encounters connection error, the self.reset() method will
        be triggered. If LMANC use the original init_units() method in SMAC package, the self.reset() of SMAC will be
        triggered and the randomized starting location cannot be loaded.
        """
        while True:
            # Sometimes not all units have yet been created by SC2
            self.agents = {}
            self.enemies = {}

            ally_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 1
            ]
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                reverse=False,
            )

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug:
                    logging.debug(
                        "Unit {} is {}, x = {}, y = {}".format(
                            len(self.agents),
                            self.agents[i].unit_type,
                            self.agents[i].pos.x,
                            self.agents[i].pos.y,
                        )
                    )

            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 2:
                    self.enemies[len(self.enemies)] = unit
                    if self._episode_count == 0:
                        self.max_reward += unit.health_max + unit.shield_max

            if self._episode_count == 0:
                min_unit_type = min(
                    unit.unit_type for unit in self.agents.values()
                )
                self._init_ally_unit_types(min_unit_type)

            all_agents_created = len(self.agents) == self.n_agents
            all_enemies_created = len(self.enemies) == self.n_enemies

            self._unit_types = [
                                   unit.unit_type for unit in ally_units_sorted
                               ] + [
                                   unit.unit_type
                                   for unit in self._obs.observation.raw_data.units
                                   if unit.owner == 2
                               ]

            if all_agents_created and all_enemies_created:  # all good
                return

            try:
                self._controller.step(1)
                self._obs = self._controller.observe()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
                self.reset(**self.episode_config)

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

        - agent movement features (where it can move to, height information
            and pathing grid)
        - enemy features (available_to_attack, health, relative_x, relative_y,
            shield, unit_type)
        - ally features (visible, distance, relative_x, relative_y, shield,
            unit_type)
        - agent unit features (health, shield, unit_type)

        All of this information is flattened and concatenated into a list,
        in the aforementioned order. To know the sizes of each of the
        features inside the final list of features, take a look at the
        functions ``get_obs_move_feats_size()``,
        ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
        ``get_obs_own_feats_size()``.

        The size of the observation vector may vary, depending on the
        environment configuration and type of units present in the map.
        For instance, non-Protoss units will not have shields, movement
        features may or may not include terrain height and pathing grid,
        unit_type is not included if there is only one type of unit in the
        map etc.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        unit = self.get_unit_by_id(agent_id)

        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)  # default value is 9

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]  # skip the first 2 elements, no-op and stop in avail_actions[]

            ind = self.n_actions_move  # default is 4

            if self.obs_pathing_grid:
                move_feats[
                ind: ind + self.n_obs_pathing  # default is 8
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[
                ind: ind + self.n_obs_height  # default is 9
                ] = self.get_surrounding_height(unit)
                ind += self.n_obs_height

            if self.obs_direction_command:
                move_feats[
                ind: ind + self.n_obs_direction_command
                ] = self.get_direction_command(unit)
                ind += self.n_obs_direction_command

            if self.obs_distance_target:
                move_feats[ind:] = self.get_distance_target(
                    unit) / 120  # the distance between the starting point and target.

            # Enemy features
            idx = -1
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if (
                        dist < sight_range and e_unit.health > 0
                ):  # visible and alive
                    # Sight range > shoot range
                    if self.only_local_enemy_obs:
                        idx += 1
                        if idx == self.n_local_enemies:
                            break
                    else:
                        idx = e_id
                    enemy_feats[idx, 0] = avail_actions[
                        self.n_actions_no_attack + idx
                        ]  # available
                    enemy_feats[idx, 1] = dist / sight_range  # distance
                    enemy_feats[idx, 2] = (
                                                  e_x - x
                                          ) / sight_range  # relative X
                    enemy_feats[idx, 3] = (
                                                  e_y - y
                                          ) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[idx, ind] = (
                                e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[idx, ind] = (
                                    e_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[idx, ind + type_id] = 1  # unit type

            # Ally features
            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                if (
                        dist < sight_range and al_unit.health > 0
                ):  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                                al_unit.health / al_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (
                                    al_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1

        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )

        if self.obs_timestep_number:
            agent_obs = np.append(
                agent_obs, self._episode_steps / self.episode_limit)

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug("Avail. actions {}".format(
                self.get_avail_agent_actions(agent_id)))
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))

        return agent_obs

    def get_obs_ally_feats_size(self):
        """Returns the dimensions of the matrix containing allied features.
        Size is n_allies x n_features.
        In hierarchical control, agents could observe all allied agents
        or only obs the agents within the same team.
        """
        nf_al = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        if self.obs_last_action:
            nf_al += self.n_actions

        if self.hierarchical:
            return self.n_ally_agent_in_platoon - 1, nf_al
        else:
            return self.n_agents - 1, nf_al

    def get_obs_enemy_feats_size(self, local=0):
        """ Returns the dimensions of the matrix containing enemy features.
        Size is n_enemies x n_features.
        In LMANC we could restrict the size of obs of enemies by only include
        nearby enemies.
        """
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        if local == 1:
            return self.n_local_enemies, nf_en

        return self.n_enemies, nf_en

    def unit_shoot_range(self, agent_id):
        """Returns the shooting range for an agent.
        The shoot ranges varies according to unit types in LMANC."""
        if self.get_unit_by_id(agent_id).unit_type == 1936:  # Siege Tank
            return 7
        else:
            return 6

    def unit_sight_range(self, agent_id):
        """Returns the sight range for an agent.
        The sight ranges varies according to unit types in LMANC."""
        if self.get_unit_by_id(agent_id).unit_type == 1936:  # Siege Tank
            return 11
        else:
            return 9

    def get_obs_move_feats_size(self):
        """Returns the size of the vector containing the agents'
        movement-related features.
        In LMANC, the observation of movement-related features include the
        direction and distance towards targets locations."""
        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height

        # Include the observation of command direction and the distance to targets in LMANC
        if self.obs_direction_command:
            move_feats += self.n_obs_direction_command
        if self.obs_distance_target:
            move_feats += self.n_obs_distance_target

        return move_feats

    def _launch(self):
        """Launch the StarCraft II game."""
        self._run_config = run_configs.get(version=self.game_version)
        _map = maps.get(self.map_name)  # use the input map name from the folder that contains maps.

        # Setting up the interface
        interface_options = sc_pb.InterfaceOptions(raw=True,
                                                   score=True)
        self._sc2_proc = self._run_config.start(window_size=self.window_size, want_rgb=False)  # start the environment
        self._controller = self._sc2_proc.controller

        # Request to create the game
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self._run_config.map_data(_map.path)),
            realtime=False,
            random_seed=self._seed)
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(type=sc_pb.Computer, race=races[self._bot_race],
                                difficulty=difficulties["7"])

        # create the game
        self._controller.create_game(create)

        join = sc_pb.RequestJoinGame(race=races[self._agent_race],
                                     options=interface_options)
        self._controller.join_game(join)  # join the game. now 2 teams are in the map, but frozen.

        game_info = self._controller.game_info()
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y

        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8))
            self.pathing_grid = np.transpose(np.array([
                [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                for row in vals], dtype=np.bool))
        else:
            self.pathing_grid = np.invert(np.flip(np.transpose(np.array(
                list(map_info.pathing_grid.data), dtype=np.bool).reshape(
                self.map_x, self.map_y)), axis=1))

        self.terrain_height = np.flip(
            np.transpose(np.array(list(map_info.terrain_height.data))
                         .reshape(self.map_x, self.map_y)), 1) / 255

    def get_init_pos(self):
        """Retrieve the initial starting coordination of units."""
        for i in range(len(self.agents)):
            self.initial_pos.append([self.agents[i].pos.x, self.agents[i].pos.y])
        return

    def update_max_reward(self):
        """Update the max_reward with movement rewards.

        This method is called every time when an environment is initialized in MOSMAC scenarios."""
        self.max_reward = self.n_enemies * self.reward_death_value

        for unit in self._obs.observation.raw_data.units:
            if unit.owner == 2:
                self.max_reward += unit.health_max + unit.shield_max

        self.max_reward = self.alpha * self.max_reward + \
                          (1 - self.alpha) * (self.n_agents * self.reward_SP + self.n_sp * self.reward_arrive) \
                          + self.reward_win
        return

    def reset(self, **episode_config):
        """
        Reset the environment. Required after each full episode.
        Returns initial observations and states.

        Randomize the environment with episode_config.
        """
        self.episode_config = episode_config
        if len(self.map_sps.items()) == 2:
            self.map_sps[1] = episode_config['target_location']

        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        else:
            self._restart()

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        try:
            self._obs = self._controller.observe()
            self.init_units()  # Max reward is updated inside.
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

        self.get_init_pos()  # New in LMANC.
        self.update_max_reward()  # New in LMANC
        self.init_move_record()  # New in LMANC

        # Get the randomized target location
        self.target_SP_loc = episode_config.get("target_location", {})

        if self.debug:
            logging.debug(
                "Started Episode {}".format(self._episode_count).center(
                    60, "*"
                )
            )

        if not self.hierarchical:
            return self.get_obs(), self.get_state()

        else:  # New in LMANC.
            self.death_tracker_ally = np.zeros((self.n_ally_platoons, self.n_ally_agent_in_platoon))
            self.last_action = np.zeros((self.n_ally_platoons, self.n_ally_agent_in_platoon, self.n_actions))
            try:
                self.init_platoons()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
            return self.get_obs_company()

    def init_move_record(self):
        """Track movement information in the LMANC environment.
        MOSMAC environment support multiple cooperative teams, controlled by a single
        higher-level controller.
        It also supports single team config. To do so, simply set self.n_ally_platoons as 1.
        Besides recording the movement progress, this method initializes the sub-goal.
        For scenarios without intermediate sub-goals, agents will take this goal as
        the final target."""

        self.agents_movement_record = [[None] * self.n_ally_agent_in_platoon for i in range(self.n_ally_platoons)]
        if self.map_sps is not None:
            self.platoons_move_record = [
                [0] * (len(self.map_sps) + 1)
                for _ in range(self.n_ally_platoons)
            ]
        else:  # Randomize target
            self.platoons_move_record = [
                [0] * 3
                for _ in range(self.n_ally_platoons)
            ]

        if self.hierarchical:
            self.target_SP_loc = [self.map_sps[0] for _ in range(self.n_ally_platoons)]
            self.target_SP_id = [0 for _ in range(self.n_ally_platoons)]
        else:
            self.target_SP_loc = self.map_sps[1]
            self.target_SP_id = 1

    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        Compared with the method in the superclass, this new method consider the winning
        condition in the MOSMAC task, where agents need to reach certain specific locations.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        if self.hierarchical:  # default is false
            self.previous_ally_platoons = deepcopy(self.ally_platoons)
            self.previous_enemy_units = deepcopy(self.enemies)
        else:
            self.previous_ally_units = deepcopy(self.agents)
            self.previous_enemy_units = deepcopy(self.enemies)

        if self.hierarchical:  # default is false
            for platoon in self.ally_platoons:
                for al_id, al_unit in enumerate(platoon):
                    updated = False
                    for unit in self._obs.observation.raw_data.units:
                        if al_unit.tag == unit.tag:
                            platoon[al_id] = unit
                            updated = True
                            n_ally_alive += 1
                            break
                    if not updated:  # dead
                        al_unit.health = 0
        else:
            for al_id, al_unit in self.agents.items():
                updated = False
                for unit in self._obs.observation.raw_data.units:
                    if al_unit.tag == unit.tag:
                        self.agents[al_id] = unit
                        updated = True
                        n_ally_alive += 1
                        break

                if not updated:  # dead
                    al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # dead
                e_unit.health = 0

        if (
                n_ally_alive == 0
                and n_enemy_alive > 0
                or self.only_medivac_left(ally=True)
        ):
            return -1  # lost

        # Check whether agents have reach the target
        agent_reach_point = [False for _ in range(self.n_agents)]
        for al_id, al_unit in self.agents.items():
            distance = math.sqrt(
                (al_unit.pos.x - self.target_SP_loc[0]) ** 2 +
                (al_unit.pos.y - self.target_SP_loc[1]) ** 2
            )
            if distance < 5 and al_unit.health != 0:
                agent_reach_point[al_id] = True

        number_arrive = agent_reach_point.count(True)
        if (
                number_arrive == n_ally_alive > 0 == n_enemy_alive
                or self.only_medivac_left(ally=False)
        ):
            return 1  # win

        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None

    def reward_battle(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to allied units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        if self.reward_sparse:
            return 0

        reward = 0
        delta_ally_deaths = 0
        delta_enemy_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        reward_movements = 0  # New in LMANC
        reward_subgoal = 0  # New in LMANC and SMANC
        neg_scale = self.reward_negative_scale

        reward_combat = 0
        reward_navigation = 0
        reward_safety = 0

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # has not died so far
                prev_health = (
                        self.previous_ally_units[al_id].health
                        + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_ally_deaths -= self.reward_death_value * neg_scale  # the punishment for death
                    delta_ally += prev_health * neg_scale  # the damage deal to the ally unit
                else:
                    # still alive
                    delta_ally += neg_scale * (
                            prev_health - al_unit.health - al_unit.shield
                    )

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                        self.previous_enemy_units[e_id].health
                        + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_enemy_deaths += self.reward_death_value
                    delta_enemy += prev_health  # damage dealt to the enemy
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        # Use the difference between last distance and current position, divided by the total distance
        for al_id, al_unit in self.agents.items():  # New in LMANC, calculate the reward of movements.
            initial_distance = self.distance(
                self.initial_pos[al_id][0],
                self.initial_pos[al_id][1],
                self.target_SP_loc[0],
                self.target_SP_loc[1],
            )

            last_distance = self.distance(
                self.previous_ally_units[al_id].pos.x,
                self.previous_ally_units[al_id].pos.y,
                self.target_SP_loc[0],
                self.target_SP_loc[1],
            )

            current_distance = self.distance(
                al_unit.pos.x,
                al_unit.pos.y,
                self.target_SP_loc[0],
                self.target_SP_loc[1],
            )

            distance_delta = last_distance - current_distance
            if initial_distance == 0:
                reward_movements = 0
            else:
                reward_movements += distance_delta / initial_distance * self.reward_SP

        # calculate the reward of completing sub-goals.
        if self.platoons_move_record[0][self.target_SP_id + 1] == 1:
            # Note: We don't include this reward in the MOSMAC experiments.
            #reward_subgoal = self.reward_arrive

            self.update_subgoal(
                completed_subgoal=self.target_SP_id)

        # Enable the program to track rewards for individual objectives.
        # reward for the combat objective
        reward_combat = delta_enemy + delta_enemy_deaths
        # reward for the navigation objective, the reward_subgoal is by default 0, unless change the code.
        reward_navigation = reward_movements + reward_subgoal
        # the reward for safety is a negative value. It is a penalty for being damaged by enemies.
        reward_safety = delta_ally_deaths - delta_ally

        if not self.momarl_setting:
            if self.reward_only_positive:  # Change the superclass's calculation of total reward of a single step.
                # The new reward a weighted avg of origin reward and reward of movements.
                reward = self.alpha * abs(reward_combat) + \
                         (1 - self.alpha) * reward_navigation
                return reward, [reward_combat, reward_navigation]
            else:
                reward = self.alpha * (reward_combat + reward_safety) + \
                         (1 - self.alpha) * (reward_navigation)
                return reward, [reward_combat, reward_navigation]
        else:
            if self.reward_only_positive and self.reward_objectives == ['reward_combat', 'reward_navigate']:
                rewards = np.array([reward_combat, reward_navigation])
                weights = np.array([self.alpha, 1 - self.alpha])
                reward = np.dot(rewards, weights)
                return reward, [reward_combat, reward_navigation]
            elif self.reward_objectives == ['reward_safety', 'reward_navigate']:
                rewards = np.array([reward_safety, reward_navigation])
                weights = np.array([self.alpha, 1 - self.alpha])
                reward = np.dot(rewards, weights)
                return reward, [reward_safety, reward_navigation]
            elif self.reward_objectives == ['reward_combat', 'reward_safety', 'reward_navigate']:
                rewards = np.array([reward_combat, reward_safety, reward_navigation])
                weights = np.array([1/3, 1/3, 1/3])
                reward = np.dot(rewards, weights)
                return reward, [reward_combat, reward_safety, reward_navigation]

    def step(self, actions):
        """
        Executes a set of actions in the environment and advances the game state by one time step.

        Parameters
        ----------
        actions : list
            A list of actions, where each action is an integer corresponding to the action that each agent should take.

        Returns
        -------
        tuple
            A tuple containing the following elements:

            reward: list
                An aggregated single scalar reward for all agents after the actions have been executed.
            terminated: bool
                A boolean flag indicating whether the episode has ended.
            info: dict
                A dictionary containing diagnostic information useful for debugging.
            reward_objs: list
                A list of rewards for individual objectives.
                The length of it should be equaled to the number of objectives.

        Notes
        -----
        This method overrides the original implementation in EPyMARL and PyMARL.
        In the original implementation, this method returns a scalar reward, terminated, and info.
        In the MOSMAC environment, we enable the program to track rewards for individual objectives.
        """
        actions_int = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for a_id, action in enumerate(actions_int):
            if not self.heuristic_ai:
                sc_action = self.get_agent_action(a_id, action)
            else:
                sc_action, action_num = self.get_agent_action_heuristic(
                    a_id, action
                )
                actions[a_id] = action_num
            if sc_action:
                sc_actions.append(sc_action)

        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)
        try:
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self._controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            if not self.momarl_setting:
                return 0, True, {}
            else:
                # We need to return the reward_objs in the momarl setting
                if len(self.reward_objectives) == 2:
                    return 0, True, {"reward_objective_1": 0, "reward_objective_2": 0}
                if len(self.reward_objectives) == 3:
                    return 0, True, {"reward_objective_1": 0, "reward_objective_2": 0, "reward_objective_3": 0}
        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()

        terminated = False
        if not self.momarl_setting:
            reward = self.reward_battle()
        else:
            reward, reward_objs = self.reward_battle()  # the reward_objs should also be normalized

        info = {"battle_won": False}

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0
        for _al_id, al_unit in self.agents.items():
            if al_unit.health == 0:
                dead_allies += 1
        for _e_id, e_unit in self.enemies.items():
            if e_unit.health == 0:
                dead_enemies += 1

        info["dead_allies"] = dead_allies
        info["dead_enemies"] = dead_enemies

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, "-"))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate

        self.reward = reward

        info["reward_objective_1"] = reward_objs[0]
        info["reward_objective_2"] = reward_objs[1]
        if len(reward_objs) >= 3:
            info["reward_objective_3"] = reward_objs[2]

        return reward, terminated, info

class StarCraft2EnvSMANC(StarCraft2EnvLMANC):
    """
    The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios on various paths.

    It inherits the methods of StarCraft2EnvLMANC and StarCraft2Env.
    Modifications are made by Shubham Pateria.
    Integrations are made by @Minghong GENG.
    This is the most mature environment as it inherit all method of the LMANC environment, and includes some additional
    methods for random path selection and sub-goal related functions.

    """

    def update_units(self):
        """Update units after an environment step. This function assumes that self._obs is up-to-date.

        It overrides the method in LMANC, and the origin StarCraft2Env by considering the winning condition in the \
        long-horizon tasks.
        Specifically, this updated method considers the rewards for occupying strategic locations.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        if self.hierarchical:  # default is false
            self.previous_ally_platoons = deepcopy(self.ally_platoons)
            self.previous_enemy_units = deepcopy(self.enemies)
        else:
            self.previous_ally_units = deepcopy(self.agents)
            self.previous_enemy_units = deepcopy(self.enemies)

        if self.hierarchical:  # default is false
            for platoon in self.ally_platoons:
                for al_id, al_unit in enumerate(platoon):
                    updated = False
                    for unit in self._obs.observation.raw_data.units:
                        if al_unit.tag == unit.tag:
                            platoon[al_id] = unit
                            updated = True
                            n_ally_alive += 1
                            break
                    if not updated:  # dead
                        al_unit.health = 0
        else:
            for al_id, al_unit in self.agents.items():
                updated = False
                for unit in self._obs.observation.raw_data.units:
                    if al_unit.tag == unit.tag:
                        self.agents[al_id] = unit
                        updated = True
                        n_ally_alive += 1
                        break

                if not updated:  # dead
                    al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # dead
                e_unit.health = 0

        if (
                n_ally_alive == 0
                and n_enemy_alive > 0
                or self.only_medivac_left(ally=True)
        ):
            return -1  # lost

        complete_subgoal = self.if_complete_subgoal(
            n_ally_alive=n_ally_alive,
            n_enemy_alive=n_enemy_alive,
        )

        if complete_subgoal and self.target_SP_id == self.final_target_index:
            complete_final_goal = True
        else:
            complete_final_goal = False

        if complete_subgoal:
            if complete_final_goal:
                # agents have reach the final target.
                return 1  # win # If agent reach the last sp and win the game, we don't award the reward of reach sp to it. this reward will be overridden by the reward for winning.
            else:
                self.update_move_record(
                    completed_subgoal=self.target_SP_id,
                )  # Update the subgoal after calculating the reward.
                return None

        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None

    def init_R(self):
        """
        Create an adjacency matrix, which shows the available paths for any strategic point.
        """
        points_list = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 6),
            (6, 11),
            (11, 12),
            (1, 5),
            (5, 6),
            (0, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11)
        ]
        goal = 12

        matrix = np.matrix(
            np.ones(
                shape=(
                    len(self.map_sps),
                    len(self.map_sps),
                )
            )
        )

        matrix *= -1
        for path in points_list:
            matrix[path] = 1
        matrix[goal, goal] = 1

        return matrix

    def get_target_sp_id(self, current_location: int):
        """Update targets based on agent's current location in terms of strategic point."""
        current_location_idx = self.current_path.index(
            current_location
        )
        if current_location_idx != len(self.current_path) - 1:  # if agents haven't reached the final target.
            target_sp = self.current_path[
                current_location_idx + 1
                ]
        else:
            target_sp = current_location

        return target_sp

    def if_complete_subgoal(self, n_ally_alive: int, n_enemy_alive: int) -> bool:
        """ Check whether agents have reach the prescribed target.

        If agents arrived the prescribed target, they will be provided with a new target."""
        complete_subgoal = [0 for _ in range(self.n_agents)]
        n_agent_complete = 0

        for al_id, al_unit in self.agents.items():
            distance_to_goal = self.distance(
                al_unit.pos.x,
                al_unit.pos.y,
                self.target_SP_loc[0],
                self.target_SP_loc[1],
            )

            if distance_to_goal < 5 and al_unit.health != 0:
                complete_subgoal[al_id] = True  # Agent has reached the target
                n_agent_complete += 1

        if n_agent_complete == n_ally_alive:
            return True
        else:  # didn't complete
            return False

    def update_subgoal(self, completed_subgoal: int):
        """ Update the targets for agents.

        Parameters
        ----------
        completed_subgoal : int
            The index of the target that was just completed. This is used to determine the next target for the agents.

        Notes
        -----
        This method updates the target strategic point (SP) ID and location based on the completed subgoal.
        It also prints out the current and next target SPs for tracking purposes."""
        self.target_SP_id = self.get_target_sp_id(completed_subgoal)
        self.target_SP = self.target_SP_id
        self.target_SP_loc = self.map_sps[self.target_SP]

        print('The current target is SP {}. The next target is SP {}.'.format(completed_subgoal, self.target_SP_id))

    def update_move_record(self, completed_subgoal: int):
        """Update the movement record of allied agents.


        Parameters
        ----------
        completed_subgoal : int
            The index of the sub-goal which is completed in the last step.

        Notes
        ---------
        This method is split from the update_subgoal method.
        It is because the update of movement record should take place before the calculation of reward of movement.
        Meanwhile, the update of the index and the location of the subgoal should happen after this calculation.
        """
        index_arrived_subgoal = completed_subgoal + 1  # the first digit is a special "dead" state.
        self.platoons_move_record[0][index_arrived_subgoal] = 1

    def select_path(self):
        """
        Select a sequence of tasks for allied agents.
        """
        self.current_path = self.path_sequences[self.path_id]
        self.path_id += 1
        if self.path_id == len(self.path_sequences):
            self.path_id = 0
        print('Ally path: ', self.current_path)

    def reset(self, **kwargs):
        """Reset the environment. Required after each full episode. Returns initial observations and states.

        Parameters
        ----------
        episode_limit : itr
            The maximum steps that agents are allowed to take in an episode."""
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        else:
            self._restart()

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        try:
            self._obs = self._controller.observe()
            self.init_units()  # Max reward is updated inside.
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

        self.get_init_pos()  # New in LMANC
        self.update_max_reward()  # New in LMANC
        self.init_move_record()  # New in LMANC
        self.current_path = kwargs.get('path_sequences', None)
        self.final_target_index = kwargs.get('final_target_index', None)

        if self.debug:
            logging.debug(
                "Started Episode {}".format(self._episode_count).center(
                    60, "*"
                )
            )

        if not self.hierarchical:
            return self.get_obs(), self.get_state()

        else:  # New in LMANC
            self.death_tracker_ally = np.zeros((self.n_ally_platoons, self.n_ally_agent_in_platoon))
            self.last_action = np.zeros((self.n_ally_platoons, self.n_ally_agent_in_platoon, self.n_actions))
            try:
                self.init_platoons()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
            return self.get_obs_company()
