import os
from gym.envs.registration import register

d = os.path.dirname

MAP_DIR = os.path.join(
    d(d(d(os.path.abspath(__file__)))),
    'maps'
)


register(
    id='NavGymRewardRandomize-v0',
    kwargs={
        'robot_type': 'keti',
        'time_step': 0.2,
        'min_turning_radius': 0., # 0.33712, # wheel_base / 2
        'distance_threshold': 0.5, # less then it -> reach goal
        'num_scan_stack': 1,
        'linvel_range': [0, 0.5],
        'rotvel_range': [-0.64, 0.64],
        'human_v_pref_range': [0., 0.6],
        'human_has_legs_ratio': 0.5, # modeling human leg movement
        'indoor_ratio': 0.5,
        'min_goal_dist': 10, 
        'max_goal_dist': 20, 
        'reward_param_range': dict(
            crash=([0, 0.2], 'float'),
            progress=([0, 0.2], 'float'),
            forward=([0., 0.2], 'float'),
            rotation=([0., 0.2], 'float'),
            discomfort=([0., 0.2], 'float'),
        ),
        'env_param_range': dict(
            num_humans=([5, 15], 'int'),
            # indoor map param
            corridor_width=([3, 4], 'int'),
            iterations=([80, 150], 'int'),
            # outdoor map param
            obstacle_number=([10, 10], 'int'),
            obstacle_width=([0.3, 1.0], 'float'),
            scan_noise_std=([0., 0.05], 'float'),
        ),
    },
    entry_point='hrl_nav.reward_randomize_env:NavGymRewardRandomizeEnv'
)


register(
    id='NavGymHierarchical-v0',
    kwargs={
        'robot_type': 'keti',
        'time_step': 0.2,
        'min_turning_radius': 0., # 0.33712, # wheel_base / 2
        'distance_threshold': 0.5, # less then it -> reach goal
        'num_scan_stack': 1,
        'linvel_range': [0, 0.5],
        'rotvel_range': [-0.64, 0.64],
        'human_v_pref_range': [0., 0.6],
        'human_has_legs_ratio': 0.5, # modeling human leg movement
        'indoor_ratio': 0.5,
        'min_goal_dist': 10, 
        'max_goal_dist': 20, 
        'reward_param_range': dict(
            crash=([0, 0.2], 'float'),
            progress=([0, 0.2], 'float'),
            forward=([0., 0.2], 'float'),
            rotation=([0., 0.2], 'float'),
            discomfort=([0., 0.2], 'float'),
        ),
        'env_param_range': dict(
            num_humans=([5, 15], 'int'),
            # indoor map param
            corridor_width=([3, 4], 'int'),
            iterations=([80, 150], 'int'),
            # outdoor map param
            obstacle_number=([10, 10], 'int'),
            obstacle_width=([0.3, 1.0], 'float'),
            scan_noise_std=([0., 0.05], 'float'),
        ),
    },
    entry_point='hrl_nav.hierarchical_env:NavGymHierarchicalEnv'
)
