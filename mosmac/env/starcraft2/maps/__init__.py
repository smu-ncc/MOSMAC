from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mosmac.env.starcraft2.maps import smanc_maps


def get_map_params(map_name):
    map_param_registry = smanc_maps.get_smac_map_registry()
    return map_param_registry[map_name]
