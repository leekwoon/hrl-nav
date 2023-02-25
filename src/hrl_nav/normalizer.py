import torch
import torch.nn as nn


# normalize scan, vel, polar coord of goal
class HighLevelNormalizer(nn.Module):
    def __init__(
        self, 
        num_scan_stack,
        range_max, # max scan range
        n_angles, # scan n_angles
        max_goal_dist,
        linvel_range,
        rotvel_range,
    ):
        super(HighLevelNormalizer, self).__init__()

        self.num_scan_stack = num_scan_stack
        self.range_max = range_max
        self.n_angles = n_angles
        self.max_goal_dist = max_goal_dist
        self.linvel_range = linvel_range
        self.rotvel_range = rotvel_range

    def forward(self, preprocessed_obs):
        # normalize scan to [-0.5, 0.5]
        scan = preprocessed_obs[:, :-4]
        scan = scan.view(scan.shape[0], self.num_scan_stack, self.n_angles)
        scan = torch.clamp(scan, 0, self.range_max)
        scan = scan / self.range_max - 0.5
        preprocessed_obs[:, :-4] = scan.view(scan.shape[0], -1)

        # normalize goal distance to [-0.5, 0.5]
        other = preprocessed_obs[:, -4:]
        goal_distance, linvel, rotvel = other[:, 0], other[:, 2], other[:, 3]
        preprocessed_obs[:, -4] = goal_distance / self.max_goal_dist - 0.5
        preprocessed_obs[:, -2] = (linvel - self.linvel_range[0]) / (self.linvel_range[1] - self.linvel_range[0]) - 0.5
        preprocessed_obs[:, -1] = (rotvel - self.rotvel_range[0]) / (self.rotvel_range[1] - self.rotvel_range[0]) - 0.5
        return preprocessed_obs


# normalize scan, vel, polar coord of goal, reward params
class LowLevelNormalizer(nn.Module):
    def __init__(
        self, 
        num_scan_stack,
        range_max, # max scan range
        n_angles, # scan n_angles
        max_goal_dist,
        linvel_range,
        rotvel_range,
        reward_param_range,
    ):
        super(LowLevelNormalizer, self).__init__()

        self.num_scan_stack = num_scan_stack
        self.range_max = range_max
        self.n_angles = n_angles
        self.max_goal_dist = max_goal_dist
        self.linvel_range = linvel_range
        self.rotvel_range = rotvel_range
        self.reward_param_range = reward_param_range

    def forward(self, preprocessed_obs):
        # normalize scan to [-0.5, 0.5]
        scan = preprocessed_obs[:, :-9]
        scan = scan.view(scan.shape[0], self.num_scan_stack, self.n_angles)
        scan = torch.clamp(scan, 0, self.range_max)
        scan = scan / self.range_max - 0.5
        preprocessed_obs[:, :-9] = scan.view(scan.shape[0], -1)

        # normalize goal distance to [-0.5, 0.5]
        other = preprocessed_obs[:, -9:]
        goal_distance, linvel, rotvel = other[:, 0], other[:, 2], other[:, 3]
        preprocessed_obs[:, -9] = goal_distance / self.max_goal_dist - 0.5
        preprocessed_obs[:, -7] = (linvel - self.linvel_range[0]) / (self.linvel_range[1] - self.linvel_range[0]) - 0.5
        preprocessed_obs[:, -6] = (rotvel - self.rotvel_range[0]) / (self.rotvel_range[1] - self.rotvel_range[0]) - 0.5

        # normalize reward param
        preprocessed_obs[:, -5] = (preprocessed_obs[:, -5] - self.reward_param_range['crash'][0][0]) / (self.reward_param_range['crash'][0][1] - self.reward_param_range['crash'][0][0]) - 0.5
        preprocessed_obs[:, -4] = (preprocessed_obs[:, -4] - self.reward_param_range['progress'][0][0]) / (self.reward_param_range['progress'][0][1] - self.reward_param_range['crash'][0][0]) - 0.5
        preprocessed_obs[:, -3] = (preprocessed_obs[:, -3] - self.reward_param_range['forward'][0][0]) / (self.reward_param_range['forward'][0][1] - self.reward_param_range['crash'][0][0]) - 0.5
        preprocessed_obs[:, -2] = (preprocessed_obs[:, -2] - self.reward_param_range['rotation'][0][0]) / (self.reward_param_range['rotation'][0][1] - self.reward_param_range['crash'][0][0]) - 0.5
        preprocessed_obs[:, -1] = (preprocessed_obs[:, -1] - self.reward_param_range['discomfort'][0][0]) / (self.reward_param_range['discomfort'][0][1] - self.reward_param_range['crash'][0][0]) - 0.5
        return preprocessed_obs


