# -*- coding: utf-8 -*-
import numpy as np
import gym
from gym import utils, spaces

from nav_gym_env.env import NavGymEnv


class NavGymRewardRandomizeEnv(gym.Env, utils.EzPickle):
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
        reward_param_range
    ):
        super(NavGymRewardRandomizeEnv, self).__init__()

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
            reward_param_range
        )
        self.reward_param_range = reward_param_range

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

        # n_angles + 7 에서 n_angles + 7 + 5 로 수정! (5: success를 제외한 나머지 reward params)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(self._wrapped_env.num_scan_stack * self._wrapped_env.robot.n_angles + 7 + 5,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        })
        self.action_space = self._wrapped_env.action_space

    def _sample_reward_param(self):
        param = dict()
        for key, value in self.reward_param_range.items(): 
            if value[1] == 'int':
                param[key] = np.random.choice(
                    np.arange(value[0][0], value[0][1] + 1)
                )
            elif value[1] == 'float':
                param[key] = np.random.uniform(value[0][0], value[0][1])
            else:
                raise NotImplementedError       
        return param

    def _add_reward_param_observation(self, obs):
        reward_param_observation = np.array([
            self.reward_param['crash'],
            self.reward_param['progress'],
            self.reward_param['forward'],
            self.reward_param['rotation'],
            self.reward_param['discomfort']
        ])
        obs['observation'] = np.concatenate([
            obs['observation'],
            reward_param_observation
        ])
        return obs

    # used in HER
    def compute_terminals(self, obs):
        return self._wrapped_env.compute_terminals(obs)

    def compute_reward(self, action, obs, make_render_reward_txt=False):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs, make_render_reward_txt)[0]

    def compute_rewards(self, actions, obs, make_render_reward_txt=False):
        """batch computation for fast batch sampling in HER
        10x times faster with batch_size>2000 !
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
        reward_param = observations_dict['reward_param']
        reward_crash_param = reward_param[:, 0]
        reward_progress_param = reward_param[:, 1]
        reward_forward_param = reward_param[:, 2]
        reward_rotation_param = reward_param[:, 3]
        reward_discomfort_param = reward_param[:, 4]

        distance = np.linalg.norm(desired_goals - pose, axis=1)
        prev_distance = np.linalg.norm(desired_goals - prev_pose, axis=1)

        success = (distance < self._wrapped_env.distance_threshold).astype(np.float32)
        crash = scan - self._wrapped_env.scan_threshold
        crash = np.any(crash < 0, axis=1).astype(np.float32)
        discomfort = scan - self._wrapped_env.scan_discomfort_threshold
        discomfort = np.any(discomfort < 0, axis=1).astype(np.float32)
        discomfort = np.logical_and(discomfort, np.logical_not(crash))

        batch_size = observations.shape[0]
        reward_success = np.zeros(batch_size)
        reward_crash = np.zeros(batch_size)
        reward_progress = np.zeros(batch_size)
        reward_forward = np.zeros(batch_size)
        reward_rotation = np.zeros(batch_size)
        reward_discomfort = np.zeros(batch_size)

        reward_success[success.astype(np.bool)] = 1.0 * self._wrapped_env.reward_scale
        reward_crash[crash.astype(np.bool)] = -1.0 * reward_crash_param[crash.astype(np.bool)] * self._wrapped_env.reward_scale
        reward_progress = (prev_distance - distance) * reward_progress_param * self._wrapped_env.reward_scale
        reward_forward = vel[:, 0] * reward_forward_param * self._wrapped_env.reward_scale
        # reward_rotation = -np.abs(vel[:, 1]) * reward_rotation_param
        reward_rotation = -1.0 * (vel[:, 1] ** 2) * reward_rotation_param * self._wrapped_env.reward_scale
        reward_discomfort[discomfort.astype(np.bool)] = -(1.0 - np.min(
            np.divide(
                scan[discomfort.astype(np.bool)] - self._wrapped_env.scan_threshold,
                self._wrapped_env.scan_discomfort_threshold - self._wrapped_env.scan_threshold + 1e-6
            ),
            axis=1
        )) * reward_discomfort_param[discomfort.astype(np.bool)] * self._wrapped_env.reward_scale

        reward = (
            reward_success \
            + reward_crash \
            + reward_progress \
            + reward_forward \
            + reward_rotation \
            + reward_discomfort
        )

        if make_render_reward_txt:
            self._wrapped_env._make_render_reward_txt(
                reward_success.mean(),
                reward_crash.mean(),
                reward_progress.mean(),
                reward_forward.mean(),
                reward_rotation.mean(),
                reward_discomfort.mean()
            )
        return reward # batch!

    def reset(self):
        self.reward_param = self._sample_reward_param()
        obs = self._wrapped_env.reset()
        obs = self._add_reward_param_observation(obs)
        return obs

    def step(self, action):
        obs, _, done, info = self._wrapped_env.step(action)
        obs = self._add_reward_param_observation(obs)
        reward = self.compute_reward(action, obs, make_render_reward_txt=True)

        observation_dict = observation_to_dict(
            obs['observation'],
            num_scan_stack=self._wrapped_env.num_scan_stack,
            n_angles=self._wrapped_env.robot.n_angles
        )
        reward_param = observation_dict['reward_param']
        self._wrapped_env.render_obs_txt += \
            '\nreward_param: ({:.2f} {:.2f}, {:.2f}, {:.2f}, {:.2f})\n'.format(
                reward_param[0], reward_param[1], reward_param[2], reward_param[3], reward_param[4])

        return obs, reward, done, info

    def render(self, mode='human'):
        return self._wrapped_env.render(mode)


def observation_to_dict(observation, num_scan_stack, n_angles):
    scan_stack = observation[:num_scan_stack * n_angles]
    scan = observation[(num_scan_stack - 1) * n_angles:num_scan_stack * n_angles]
    other = observation[num_scan_stack * n_angles:]
    prev_pose = other[:2]
    pose = other[2:4]
    vel = other[4:6]
    yaw = other[6]
    reward_param = other[7:]
    return dict(
        scan_stack=scan_stack,
        scan=scan,
        prev_pose=prev_pose,
        pose=pose,
        vel=vel,
        yaw=yaw,
        reward_param=reward_param
    )


def observation_batch_to_dict(observation, num_scan_stack, n_angles):
    scan_stack = observation[:, :num_scan_stack * n_angles]
    scan = observation[:, (num_scan_stack - 1) * n_angles:num_scan_stack * n_angles]
    other = observation[:, num_scan_stack * n_angles:]
    prev_pose = other[:, :2]
    pose = other[:, 2:4]
    vel = other[:, 4:6]
    yaw = other[:, 6]
    reward_param = other[:, 7:]
    return dict(
        scan_stack=scan_stack,
        scan=scan,
        prev_pose=prev_pose,
        pose=pose,
        vel=vel,
        yaw=yaw,
        reward_param=reward_param
    )

