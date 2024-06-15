from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib


class LMANCMap(lib.Map):
    directory = "LMANC_Maps"
    download = None  # todo: include the link for downloading the files here.
    players = 2
    step_mul = 8
    game_steps_per_episode = 0


map_param_registry = { }


def get_smac_map_registry():
    return map_param_registry


for name in map_param_registry.keys():
    globals()[name] = type(name, (LMANCMap,), dict(filename=name))  # check this link in smac package
