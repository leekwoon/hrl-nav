import os
import gym
import time
import random
import argparse
import numpy as np

import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.her.her import HERTrainer
from rlkit.samplers.data_collector import GoalConditionedPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import hrl_nav
from hrl_nav.normalizer import LowLevelNormalizer, HighLevelNormalizer
from hrl_nav.hierarchical_networks import (
    HighLevelConvQf, 
    HighLevelConvTanhGaussianPolicy, 
    LowLevelConvQf, 
    LowLevelConvTanhGaussianPolicy
)



# params = torch.load('/home/leekwoon/Downloads/high_itr_80.pkl',
    # params = torch.load('/home/leekwoon/Downloads/high_params.pkl',
        # map_location=torch.device('cpu'))



variant = dict(
    env_id='NavGymHierarchical-v0',
    base_logdir='/tmp',
    seed=0,
    snapshot_mode='gap_and_last',
    snapshot_gap=1, # TODO: change it!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: change it!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: change it!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    log_time=time.strftime('%Y_%m_%d_%H_%M_%S'),
    # args=dict(
    #     env_id='Dxp2DHierarchical-v0',
    #     base_logdir='/data2/seongun/',
    #     log_dir = '{}/vcrl_logs/Dxp2D-v0/hrl_default/{}/seed_{}/high_level', # base_logdir, spec, seed
    #     snapshot_mode='gap_and_last',
    #     snapshot_gap=20,
    #     seed=None,
    #     log_time=log_time,
    # ),

    # ===
        # CHECK IT
        # CHECK IT
        # CHECK IT
        # CHECK IT
        # CHECK IT
        # CHECK IT
        # CHECK IT
        # CHECK IT
    # ===
    replay_buffer_kwargs=dict(
        # TODO: comment out after debuging
        max_size=int(5E3), 
        # max_size=int(5E6), 
        fraction_goals_rollout_goals=0.2,
        fraction_goals_env_goals=0,
    ),
    sac_trainer_kwargs=dict(
        discount=0.99,
        policy_lr=3e-4,
        qf_lr=3e-4,
        soft_target_tau=5e-3,
        target_update_period=1,
        use_automatic_entropy_tuning=True,
    ), 
    algo_kwargs=dict(
        batch_size=256,
        num_epochs=1000,
        num_eval_steps_per_epoch=0,
        num_expl_steps_per_train_loop=5000,
        num_trains_per_train_loop=1000,
        min_num_steps_before_training=5000,
        max_path_length=1000,
    ),
    # type_to_func=dict(
    #     normalizer=DxpNormalizer,
    #     policy=DxpHighLevelConvTanhGaussianPolicy,
    #     qf=DxpHighLevelConvQf,
    #     # path_collector=HighLevelGoalConditionedPathCollector,
    #     path_collector=GoalConditionedPathCollector,
    # ),
)

if variant['seed'] is None:
    variant['seed'] = random.randint(0, 100000)
set_seed(variant['seed'])

logdir = '{}/hrl_nav_logs/{}/hrl/{}/seed_{}/high_level'
logdir = logdir.format(variant['base_logdir'], variant['env_id'], variant['log_time'], variant['seed'])

setup_logger(
    variant=variant, 
    log_dir=logdir,
    snapshot_mode=variant['snapshot_mode'],
    snapshot_gap=variant['snapshot_gap']
)

if torch.cuda.is_available():
    ptu.set_gpu_mode(True)

# expl_env = NormalizedBoxEnv(gym.make(variant['env_id']))
# eval_env = NormalizedBoxEnv(gym.make(variant['env_id']))

# observation_key = 'observation'
# desired_goal_key = 'desired_goal'
# achieved_goal_key = 'achieved_goal'
# scan_dim = expl_env.observation_space.spaces[observation_key].low.size - 7
# obs_dim = scan_dim + 2 + 2 # (scan_dim + goal_dim + vel_dim)
# action_dim = expl_env.action_space.low.size

low_level_agent = torch.load('/tmp/save_low/params.pkl')['evaluation/policy']

# expl_env._wrapped_env.set_low_level_agent(low_level_agent)
# eval_env._wrapped_env.set_low_level_agent(low_level_agent)

from rlkit.samplers.rollout_functions import multitask_rollout
from nav_gym_env.ros_env import RosEnv
from hrl_nav.scenario_wrapper import ScenarioWrapper

env = gym.make('NavGymHierarchical-v0')
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
env = NormalizedBoxEnv(env)
env = RosEnv(env)
env._wrapped_env.set_low_level_agent(low_level_agent)

params = torch.load('/tmp/save_high/params.pkl',
# params = torch.load('/home/leekwoon/Downloads/high_params.pkl',
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

# # high lvel
# obs_normalizer = HighLevelNormalizer(
#     num_scan_stack=1,
#     range_max=6.0,
#     n_angles=expl_env._wrapped_env._wrapped_env.robot.n_angles,
#     max_goal_dist=20,
#     linvel_range=expl_env._wrapped_env._wrapped_env.linvel_range,
#     rotvel_range=expl_env._wrapped_env._wrapped_env.rotvel_range,
# )
# qf1 = HighLevelConvQf(
#     num_scan_stack=1,
#     obs_normalizer=obs_normalizer
# )
# qf2 = HighLevelConvQf(
#     num_scan_stack=1,
#     obs_normalizer=obs_normalizer
# )
# target_qf1 = HighLevelConvQf(
#     num_scan_stack=1,
#     obs_normalizer=obs_normalizer
# )
# target_qf2 = HighLevelConvQf(
#     num_scan_stack=1,
#     obs_normalizer=obs_normalizer
# )
# expl_policy = HighLevelConvTanhGaussianPolicy(
#     num_scan_stack=1,
#     obs_normalizer=obs_normalizer
# )
# eval_policy = MakeDeterministic(expl_policy)

# replay_buffer = ObsDictRelabelingBuffer(
#     env=expl_env,
#     observation_key=observation_key,
#     desired_goal_key=desired_goal_key,
#     achieved_goal_key=achieved_goal_key,
#     **variant['replay_buffer_kwargs']
# )
# # To reduce the storage space for scan data, the observation is stored as np.float16
# replay_buffer._obs['observation'] = np.zeros(
#     (replay_buffer.max_size, replay_buffer.ob_spaces['observation'].low.size), dtype=np.float16)
# replay_buffer._next_obs['observation'] = np.zeros(
#     (replay_buffer.max_size, replay_buffer.ob_spaces['observation'].low.size), dtype=np.float16)


# trainer = SACTrainer(
#     env=expl_env,
#     policy=expl_policy,
#     qf1=qf1,
#     qf2=qf2,
#     target_qf1=target_qf1,
#     target_qf2=target_qf2,
#     **variant['sac_trainer_kwargs']
# )
# trainer = HERTrainer(trainer)

# expl_path_collector = GoalConditionedPathCollector(
#     expl_env,
#     expl_policy,
#     observation_key=observation_key,
#     desired_goal_key=desired_goal_key,
# )
# eval_path_collector = GoalConditionedPathCollector(
#     eval_env,
#     eval_policy,
#     observation_key=observation_key,
#     desired_goal_key=desired_goal_key,
# )

# algorithm = TorchBatchRLAlgorithm(
#     trainer=trainer,
#     exploration_env=expl_env,
#     evaluation_env=eval_env,
#     exploration_data_collector=expl_path_collector,
#     evaluation_data_collector=eval_path_collector,
#     replay_buffer=replay_buffer,
#     **variant['algo_kwargs']
# )
# algorithm.to(ptu.device)
# algorithm._end_epoch(0)
# # algorithm.train()