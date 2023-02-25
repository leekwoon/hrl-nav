import torch
import torch.nn as nn
from torch.nn import functional as F

from rlkit.torch.distributions import TanhNormal
from rlkit.torch.sac.policies.base import TorchStochasticPolicy
from rlkit.torch.sac.policies.gaussian_policy import LOG_SIG_MIN, LOG_SIG_MAX


class HighLevelConvQf(nn.Module):
    def __init__(
        self,
        num_scan_stack,
        obs_normalizer=None,
    ):
        super(HighLevelConvQf, self).__init__()

        self.num_scan_stack = num_scan_stack
        self.obs_normalizer = obs_normalizer
        self.n_angles = 512
        self.action_dim = 5

        self.conv1 = nn.Conv1d(in_channels=num_scan_stack, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 32, 256)
        self.fc2 = nn.Linear(256 + 2 + 2 + self.action_dim, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, obs, action):
        obs = high_level_preprocess_obs(obs)
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer(obs)        
        scan = obs[:, :-4]
        scan = scan.view(scan.shape[0], self.num_scan_stack, self.n_angles)
        other = obs[:, -4:]

        x = F.relu(self.conv1(scan))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = torch.cat([x, other, action], dim=-1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class HighLevelConvTanhGaussianPolicy(TorchStochasticPolicy):
    def __init__(
        self,
        num_scan_stack,
        obs_normalizer=None,
    ):
        super(HighLevelConvTanhGaussianPolicy, self).__init__()

        self.num_scan_stack = num_scan_stack
        self.obs_normalizer = obs_normalizer
        self.n_angles = 512
        self.action_dim = 5

        self.conv1 = nn.Conv1d(in_channels=num_scan_stack, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 32, 256)
        self.fc2 = nn.Linear(256 + 2 + 2, 128)

        self.fc_mean = nn.Linear(128, self.action_dim)
        self.fc_log_std = nn.Linear(128, self.action_dim)

    def forward(self, obs):
        obs = high_level_preprocess_obs(obs)
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer(obs)        
        scan = obs[:, :-4]
        scan = scan.view(scan.shape[0], self.num_scan_stack, self.n_angles)
        other = obs[:, -4:]

        x = F.relu(self.conv1(scan))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = torch.cat([x, other], dim=-1)
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        return TanhNormal(mean, std)


def high_level_preprocess_obs(obs):
    observation = obs[:, :-2]
    goal = obs[:, -2:]

    scan = observation[:, :-7]
    other = observation[:, -7:]
    prev_pose = other[:, :2]
    pose = other[:, 2:4]
    vel = other[:, 4:6]
    yaw = other[:, 6]

    goal_distance = (pose - goal).pow(2).sum(-1).sqrt()
    goal_angle = torch.atan2(
        goal[:, 1] - pose[:, 1], goal[:, 0] - pose[:, 0])
    # normalize it from -pi to pi
    goal_angle = torch.atan2(torch.sin(goal_angle), torch.cos(goal_angle))
    # relative angle to goal
    heading = torch.atan2(torch.sin(goal_angle - yaw), torch.cos(goal_angle - yaw))
    polar_coords = torch.cat([goal_distance[:, None], heading[:, None]], axis=1)
    preprocessed_obs = torch.cat([scan, polar_coords, vel], axis=1)
    return preprocessed_obs


class LowLevelConvQf(nn.Module):
    def __init__(
        self,
        num_scan_stack,
        obs_normalizer=None,
    ):
        super(LowLevelConvQf, self).__init__()

        self.num_scan_stack = num_scan_stack
        self.obs_normalizer = obs_normalizer
        self.n_angles = 512
        self.action_dim = 2

        self.conv1 = nn.Conv1d(in_channels=num_scan_stack, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 32, 256)
        self.fc2 = nn.Linear(256 + 2 + 2 + 5 + self.action_dim, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, obs, action):
        obs = low_level_preprocess_obs(obs)
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer(obs)
        scan = obs[:, :-4-5]
        scan = scan.view(scan.shape[0], self.num_scan_stack, self.n_angles)
        other = obs[:, -4-5:]

        x = F.relu(self.conv1(scan))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = torch.cat([x, other, action], dim=-1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LowLevelConvTanhGaussianPolicy(TorchStochasticPolicy):
    def __init__(
        self,
        num_scan_stack,
        obs_normalizer=None,
    ):
        super(LowLevelConvTanhGaussianPolicy, self).__init__()

        self.num_scan_stack = num_scan_stack
        self.obs_normalizer = obs_normalizer
        self.n_angles = 512
        self.action_dim = 2

        self.conv1 = nn.Conv1d(in_channels=num_scan_stack, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 32, 256)
        self.fc2 = nn.Linear(256 + 2 + 2 + 5, 128)

        self.fc_mean = nn.Linear(128, self.action_dim)
        self.fc_log_std = nn.Linear(128, self.action_dim)

    def forward(self, obs):
        obs = low_level_preprocess_obs(obs)
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer(obs)
        scan = obs[:, :-4-5]
        scan = scan.view(scan.shape[0], self.num_scan_stack, self.n_angles)
        other = obs[:, -4-5:]

        x = F.relu(self.conv1(scan))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = torch.cat([x, other], dim=-1)
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        return TanhNormal(mean, std)


def low_level_preprocess_obs(obs):
    observation = obs[:, :-2]
    goal = obs[:, -2:]

    scan = observation[:, :-7-5]
    other = observation[:, -7-5:]
    prev_pose = other[:, :2]
    pose = other[:, 2:4]
    vel = other[:, 4:6]
    yaw = other[:, 6]
    reward_param = other[:, 7:]

    goal_distance = (pose - goal).pow(2).sum(-1).sqrt()
    goal_angle = torch.atan2(
        goal[:, 1] - pose[:, 1], goal[:, 0] - pose[:, 0])
    # normalize it from -pi to pi
    goal_angle = torch.atan2(torch.sin(goal_angle), torch.cos(goal_angle))
    # relative angle to goal
    heading = torch.atan2(torch.sin(goal_angle - yaw), torch.cos(goal_angle - yaw))
    polar_coords = torch.cat([goal_distance[:, None], heading[:, None]], axis=1)
    preprocessed_obs = torch.cat([scan, polar_coords, vel, reward_param], axis=1)
    return preprocessed_obs


