from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # record the rewards for each individual objective
        self.test_objective_rewards = []
        self.train_objective_rewards = []

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        if len(self.env.env.reward_objectives) == 2:
            episode_reward_objs = [None, None]
            if self.args.env_args['momarl_setting'] is True:
                episode_reward_objs = [0, 0]  # Initialize accumulated rewards

        if len(self.env.env.reward_objectives) == 3:
            episode_reward_objs = [None, None, None]
            if self.args.env_args['momarl_setting'] is True:
                episode_reward_objs = [0, 0, 0]  # Initialize accumulated rewards

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            reward, terminated, env_info = self.env.step(actions[0])
            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward
            if self.args.env_args['momarl_setting'] is True:
                if "reward_objective_3" not in env_info:
                    episode_reward_objs = [a + b for a, b in zip(episode_reward_objs,
                                                                 [env_info['reward_objective_1'],
                                                                  env_info['reward_objective_2']])]
                else:
                    episode_reward_objs = [a + b for a, b in zip(episode_reward_objs,
                                                                 [env_info['reward_objective_1'],
                                                                  env_info['reward_objective_2'],
                                                                  env_info['reward_objective_3']])]
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        # record the rewards for each objective
        if self.args.env_args['momarl_setting'] is True:
            cur_objective_rewards = self.test_objective_rewards if test_mode else self.train_objective_rewards
            cur_objective_rewards.append(episode_reward_objs)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            if not self.args.env_args['momarl_setting']:
                self._log(cur_returns, cur_stats, log_prefix)
            else:
                self._log(cur_returns, cur_stats, log_prefix, cur_objective_rewards)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            if not self.args.env_args['momarl_setting']:
                self._log(cur_returns, cur_stats, log_prefix)
            else:
                self._log(cur_returns, cur_stats, log_prefix, cur_objective_rewards)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix, objective_rewards=None):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

        if objective_rewards:
            objective_1 = [sublist[0] for sublist in objective_rewards]
            objective_2 = [sublist[1] for sublist in objective_rewards]
            self.logger.log_stat(prefix + "return_objective_1", np.mean(objective_1), self.t_env)
            self.logger.log_stat(prefix + "return_objective_2", np.mean(objective_2), self.t_env)
            if len(objective_rewards[0]) == 3:
                objective_3 = [sublist[2] for sublist in objective_rewards]
                self.logger.log_stat(prefix + "return_objective_3", np.mean(objective_3), self.t_env)
            objective_rewards.clear()