import random

from mosmac.env import StarCraft2EnvLMANC
from mosmac.env import StarCraft2EnvSMANC
from mosmac.env import MultiAgentEnv


# NOTE: The writing of these wrappers are referenced from SMACv2: https://github.com/oxwhirl/smacv2

class StarCraftRandomEnvWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.env = StarCraft2EnvLMANC(**kwargs)
        self.episode_limit = self.env.episode_limit

    def reset(self):
        '''Generate a randomized target location and passed into env'''
        episode_config = {}
        target_location = self._generate_random_target_location()
        episode_config["target_location"] = target_location

        return self.env.reset(**episode_config)

    def _generate_random_target_location(self):
        '''Generate a randomized target location,
        Generate a random value between 6 and 26'''
        target_location = [random.uniform(6, 26), random.uniform(6, 26)]
        # print("target_location: ", target_location)
        return target_location

    def get_obs(self):
        return self.env.get_obs()

    def get_obs_feature_names(self):
        return self.env.get_obs_feature_names()

    def get_state(self):
        return self.env.get_state()

    def get_state_feature_names(self):
        return self.env.get_state_feature_names()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_total_actions(self):
        return self.env.get_total_actions()

    def get_capabilities(self):
        return self.env.get_capabilities()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def render(self):
        return self.env.render()

    def step(self, actions):
        return self.env.step(actions)

    def get_stats(self):
        return self.env.get_stats()

    def full_restart(self):
        return self.env.full_restart()

    def save_replay(self):
        self.env.save_replay()

    def close(self):
        return self.env.close()


class StarCraftRandomSequentialTargetWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.env = StarCraft2EnvSMANC(**kwargs)
        self.episode_limit = self.env.episode_limit
        self.fixed_sequence = self.env.fixed_sequence

    def reset(self):
        """ Generate a randomized target location and passed into env """
        episode_config = {}
        random_path = self._generate_random_path(self.fixed_sequence)
        episode_config["path_sequences"] = random_path
        if self.env.number_of_subtask == 'None':
            episode_config["final_target_index"] = random_path[-1]
        else:
            episode_config["final_target_index"] = random_path[self.env.number_of_subtask - 1]
        episode_config["episode_limit"] = self.env.episode_limit

        print("The final target for agents is to move to SP {}.".format(episode_config["final_target_index"]))
        return self.env.reset(**episode_config)

    def _generate_random_path(self, fixed_sequence):
        '''Generate a randomized path from SP1 to SP14'''
        # Define the graph as a dictionary
        graph = {
            0: [1, 6],
            1: [2, 6],
            2: [3, 7],
            3: [4],
            4: [5],
            5: [8],
            6: [9],
            7: [8],
            8: [12],
            9: [10],
            10: [11],
            11: [12],
            12: [13],
            13: [13],
        }

        # Set the start and target nodes
        start_node = 1
        target_node = 13

        # Randomly select a path from the start to the target node
        path = [start_node]
        current_node = start_node
        while current_node != target_node:
            next_nodes = graph[current_node]
            if not next_nodes:
                # The current node has no neighbors, so backtrack
                path.pop()
                current_node = path[-1]
            else:
                # Choose a random neighbor and move to it
                next_node = random.choice(next_nodes)
                path.append(next_node)
                current_node = next_node

        if fixed_sequence != 'None':
            path = fixed_sequence
        # Print the path
        # print(' -> '.join(str(path)))
        print("The selected path is:", path)

        return path

    def _generate_random_target_location(self):
        '''Generate a randomized target location,
        Generate a random value between 6 and 26'''
        target_location = [random.uniform(6, 26), random.uniform(6, 26)]
        print("target_location: ", target_location)
        return target_location

    def get_obs(self):
        return self.env.get_obs()

    def get_obs_feature_names(self):
        return self.env.get_obs_feature_names()

    def get_state(self):
        return self.env.get_state()

    def get_state_feature_names(self):
        return self.env.get_state_feature_names()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_total_actions(self):
        return self.env.get_total_actions()

    def get_capabilities(self):
        return self.env.get_capabilities()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def render(self):
        return self.env.render()

    def step(self, actions):
        return self.env.step(actions)

    def get_stats(self):
        return self.env.get_stats()

    def full_restart(self):
        return self.env.full_restart()

    def save_replay(self):
        self.env.save_replay()

    def close(self):
        return self.env.close()
