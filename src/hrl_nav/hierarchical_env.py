# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import gym
from gym import utils, spaces
import rlkit.torch.pytorch_util as ptu

from nav_gym_env.env import NavGymEnv


class NavGymHierarchicalEnv(gym.Env, utils.EzPickle):
    def __init__(
        self,
        robot_type,
        time_step,
        min_turning_radius, # to consider ackermann case.
        distance_threshold,
        num_scan_stack,
        linvel_range,
        rotvel_range,
        human_v_pref_range,
        human_has_legs_ratio,
        indoor_ratio,
        min_goal_dist,
        max_goal_dist,
        env_param_range,
        reward_param_range,
    ):
        super(NavGymHierarchicalEnv, self).__init__()

        utils.EzPickle.__init__(
            self,
            robot_type,
            time_step,
            min_turning_radius, # to consider ackermann case. 
            distance_threshold,
            num_scan_stack,
            linvel_range,
            rotvel_range,
            human_v_pref_range,
            human_has_legs_ratio,
            indoor_ratio,
            min_goal_dist,
            max_goal_dist,
            env_param_range,
            reward_param_range,
        )
        self.reward_param_range = reward_param_range
        self.low_level_agent = None

        self._wrapped_env = NavGymEnv(
            robot_type=robot_type,
            time_step=time_step,
            min_turning_radius=min_turning_radius, # wheel_base / 2
            distance_threshold=distance_threshold, # less then it -> reach goal
            num_scan_stack=num_scan_stack,
            linvel_range=linvel_range,
            rotvel_range=rotvel_range,
            human_v_pref_range=human_v_pref_range,
            human_has_legs_ratio=human_has_legs_ratio,
            indoor_ratio=indoor_ratio,
            min_goal_dist=min_goal_dist, 
            max_goal_dist=max_goal_dist, 
            env_param_range=env_param_range,
            reward_scale=15.,
            reward_success_factor=1,
            reward_crash_factor=1,
            reward_progress_factor=0.001,
            reward_forward_factor=0.0,
            reward_rotation_factor=0.005,
            reward_discomfort_factor=0.01,
        )

        self.action_space = spaces.Box(
            low=np.array([self.reward_param_range['crash'][0][0], self.reward_param_range['progress'][0][0], self.reward_param_range['forward'][0][0], self.reward_param_range['rotation'][0][0], self.reward_param_range['discomfort'][0][0]]),
            high=np.array([self.reward_param_range['crash'][0][1], self.reward_param_range['progress'][0][1], self.reward_param_range['forward'][0][1], self.reward_param_range['rotation'][0][1], self.reward_param_range['discomfort'][0][1]]),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(self._wrapped_env.num_scan_stack * self._wrapped_env.robot.n_angles + 7,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        })

    @property
    def map_info(self):
        return self._wrapped_env.map_info

    @property
    def robot(self):
        return self._wrapped_env.robot

    @property
    def humans(self):
        return self._wrapped_env.humans

    @property
    def num_scan_stack(self):
        return self._wrapped_env.num_scan_stack

    @property
    def prev_obs(self):
        return self._wrapped_env.prev_obs

    def set_low_level_agent(self, low_level_agent):
        self.low_level_agent = low_level_agent.to(ptu.device)
        print(self.low_level_agent)
        print('low_level_agent successfully loaded.')

    def compute_terminals(self, obs):
        observations = obs['observation']
        observations_dict = observation_batch_to_dict(
            observations,
            num_scan_stack=self._wrapped_env.num_scan_stack,
            n_angles=self._wrapped_env.robot.n_angles
        )
        desired_goals = obs['desired_goal']

        scan = observations_dict['scan'] 
        pose = observations_dict['pose'] 

        # done only when the robot reaches desired_goals.
        distance = np.linalg.norm(desired_goals - pose, axis=1)
        done = (distance < self._wrapped_env.distance_threshold).astype(np.float32)

        return done

    def compute_done(self, obs):
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_terminals(next_obs)[0]

    def compute_reward(self, action, obs, make_render_reward_txt=False):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs, make_render_reward_txt)[0]

    def compute_rewards(self, actions, obs, make_render_reward_txt=False):
        """
        Rewards should be sparse for the high-level environment.
        """
        observations = obs['observation']
        observations_dict = observation_batch_to_dict(
            observations,
            num_scan_stack=self._wrapped_env.num_scan_stack,
            n_angles=self._wrapped_env.robot.n_angles
        )
        desired_goals = obs['desired_goal']

        scan = observations_dict['scan'] 
        prev_pose = observations_dict['prev_pose'] 
        pose = observations_dict['pose'] 
        vel = observations_dict['vel'] 
        yaw = observations_dict['yaw'] 

        distance = np.linalg.norm(desired_goals - pose, axis=1)
        reward = -(distance >= self._wrapped_env.distance_threshold).astype(np.float32)

        if make_render_reward_txt:
            self._wrapped_env._make_render_reward_txt(
                reward.mean(),
                0.,
                0.,
                0.,
                0.,
                0.,
            )
        return reward

    def reset(self):
        obs = self._wrapped_env.reset()
        return obs

    def step(self, action):
        assert self.low_level_agent, "low_level_agent is not set. Please load low_level_agent."

        o_for_low_level_agent = self._low_level_obs_preprocessor(self._wrapped_env.prev_obs, action)
        low_level_action, _ = self.low_level_agent.get_action(o_for_low_level_agent)
        low_level_action = self._unscale_low_level_action(low_level_action)
        obs, _, _, info = self._wrapped_env.step(low_level_action)
        reward = self.compute_reward(low_level_action, obs, make_render_reward_txt=True)
        done = self.compute_done(obs)
        return obs, reward, done, info

    def render(self, mode='human'):
        return self._wrapped_env.render(mode)

    def _unscale_high_level_action(self, action):
        lb = self.action_space.low
        ub = self.action_space.high

        unscaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        return unscaled_action

    def _unscale_low_level_action(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high

        unscaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        return unscaled_action

    def _low_level_obs_preprocessor(self, o, a):
        return np.hstack((o['observation'], a, o['desired_goal']))


def observation_to_dict(observation, num_scan_stack, n_angles):
    scan_stack = observation[:num_scan_stack * n_angles]
    scan = observation[(num_scan_stack - 1) * n_angles:num_scan_stack * n_angles]
    other = observation[num_scan_stack * n_angles:]
    prev_pose = other[:2]
    pose = other[2:4]
    vel = other[4:6]
    yaw = other[6]
    return dict(
        scan_stack=scan_stack,
        scan=scan,
        prev_pose=prev_pose,
        pose=pose,
        vel=vel,
        yaw=yaw        
    )


def observation_batch_to_dict(observation, num_scan_stack, n_angles):
    scan_stack = observation[:, :num_scan_stack * n_angles]
    scan = observation[:, (num_scan_stack - 1) * n_angles:num_scan_stack * n_angles]
    other = observation[:, num_scan_stack * n_angles:]
    prev_pose = other[:, :2]
    pose = other[:, 2:4]
    vel = other[:, 4:6]
    yaw = other[:, 6]
    return dict(
        scan_stack=scan_stack,
        scan=scan,
        prev_pose=prev_pose,
        pose=pose,
        vel=vel,
        yaw=yaw        
    )