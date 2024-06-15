from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib


class MOSMACMap(lib.Map):
    directory = "MOSMAC_Maps"
    download = None  # todo: include the link for downloading the files here.
    players = 2
    step_mul = 8
    game_steps_per_episode = 0


map_param_registry = {
    "12a3p_20e5p_3pth_13sp_final": {
        "n_agents": 12,
        "n_enemies": 20,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_4t_upper_e34": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_4t_large_complex": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_4t_large_flat": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_12t_large_flat": {
        "n_agents": 4,
        "n_enemies": 12,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_12t_large_complex": {
        "n_agents": 4,
        "n_enemies": 12,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_0t_large_flat": {
        "n_agents": 4,
        "n_enemies": 0,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_0t_large_complex": {
        "n_agents": 4,
        "n_enemies": 0,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_0t_large_complex_rm_ramp": {
        "n_agents": 4,
        "n_enemies": 0,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_0t_sp1": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_0t_sp2": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_0t_sp3": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_0t_sp4": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_0t_sp5": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_0t_sp8": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_0t_sp12": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t_vs_0t_sp13": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 500,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "3t": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 50,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4t": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 50,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "8t": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 100,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "12t": {
        "n_agents": 12,
        "n_enemies": 12,
        "limit": 100,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
}


def get_smac_map_registry():
    return map_param_registry


for name in map_param_registry.keys():
    globals()[name] = type(name, (MOSMACMap,), dict(filename=name))
