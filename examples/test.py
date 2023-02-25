import os
import gym
import argparse
import numpy as np

import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.samplers.rollout_functions import multitask_rollout

from nav_gym_env.ros_env import RosEnv

import hrl_nav
from hrl_nav.scenario_wrapper import ScenarioWrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='none', type=str, choices=['none', 'corridor', 'atc', 'building'])
    parser.add_argument('--low_level_agent_params', default='', type=str)
    parser.add_argument('--high_level_agent_params', default='', type=str)
    parser.add_argument('--savedir', type=str, default='./stats/', help='save directory')
    parser.add_argument('--spec', type=str, help='save filename')
    args = parser.parse_args()

    if torch.cuda.is_available():
        ptu.set_gpu_mode(True)

    env = gym.make('NavGymHierarchical-v0')
    if args.scenario == 'corridor':
        env = ScenarioWrapper(
            env,
            map_yaml_file=os.path.join(hrl_nav.MAP_DIR, 'keti4floor.yaml'),
            resolution_factor=1.5,# high resolution factor -> wide width
            distance_threshold=1.0,
            start_xy=[10, 23.4],
            start_theta=0.,
            goal_xy=[16.2, 40],
            use_subgoal_from_global_planner=False, # True,
            subgoal_dist=6,
            scenarios=[
                [[30, 19], [10, 23.4]], # human1 start, goal
                [[28, 19], [16.2, 40]],
                [[16.2, 40], [10, 23.4]],
                [[34, 35.5], [20.7, 20.5]]
            ]
        )
    elif args.scenario == 'atc':
        env = ScenarioWrapper(
            env,
            map_yaml_file=os.path.join(hrl_nav.MAP_DIR, 'atc.yaml'),
            resolution_factor=1.2,# high resolution factor -> wide width
            distance_threshold=1.0,
            start_xy=[116.4, 24],
            start_theta=np.pi,
            goal_xy=[71, 52.4],
            use_subgoal_from_global_planner=False,
            subgoal_dist=6,
            scenarios=[
                [[84, 40.8], [124.3, 20.6]],
                [[91.6, 34.3], [132.1, 18]],
                [[89.5, 36.2], [113, 25.2]],
                [[79.8, 47.6], [127.2, 23.1]],
                [[100, 29], [127.5, 22]],
                [[89, 39.3], [120.5, 20.4]],
                [[110.2, 25], [69, 54.3]],
                [[102.4, 26.5], [59.6, 55.1]],
            ]
        )
    elif args.scenario == 'building':
        env = ScenarioWrapper(
            env,
            map_yaml_file=os.path.join(hrl_nav.MAP_DIR, 'asl_office_j.yaml'),
            resolution_factor=2.,# high resolution factor -> wide width
            distance_threshold=1.0,
            start_xy=[21.9, 16.],
            start_theta=0.,
            goal_xy=[93.3, 30.],
            use_subgoal_from_global_planner=False,
            subgoal_dist=6,
            scenarios=[
                [[20, 20], [46.6, 17.2]],
                [[42.2, 24.5], [18.1, 14.5]],
                [[53.5, 19.6], [20.6, 17.6]],
                [[66.1, 20.5], [18.1, 14]],
                [[97.1, 24.5], [42.1, 24.2]],
                [[67, 34.2], [96.8, 26.5]],
                [[41.2, 30.8], [97.7, 26]],
                [[84.2, 21.8], [41.7, 26.9]],
                [[63, 33.6], [71.3, 20.2]],
            ]
        )
    env = NormalizedBoxEnv(env)
    env = RosEnv(env)

    low_level_agent = torch.load(args.low_level_agent_params)['evaluation/policy']
    env._wrapped_env.set_low_level_agent(low_level_agent)

    params = torch.load(args.high_level_agent_params,
        map_location=torch.device('cpu'))
    policy = params['evaluation/policy']
    policy.to(ptu.device)   

    paths = []
    for i in range(10):
        print('i=', i)
        path = multitask_rollout(
            env=env,
            agent=policy,
            render=False, # True,
            render_kwargs={'mode': 'human'},
            max_path_length=1000,
            observation_key='observation',
            desired_goal_key='desired_goal',
            return_dict_obs=True,
        )
        paths.append(path)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    np.save(os.path.join(args.savedir, '{}.npy'.format(args.spec)), paths, allow_pickle=True)

