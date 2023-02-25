import os
import cv2
import gym
import numpy as np
from collections import deque
from matplotlib import pyplot as plt

import pyastar2d
import range_libc
from CMap2D import CMap2D

import nav_gym_env
from nav_gym_env.utils import (
    angle_correction,
    translation_matrix_from_xyz,
    quaternion_matrix_from_yaw,
    transform_xys
)
from nav_gym_env.human import Human
from nav_gym_env.env import xy_to_ij, batch_ij_to_xy, batch_xy_to_ij, path_to_waypoints, observation_to_dict, NavGymEnv
from hrl_nav.hierarchical_env import NavGymHierarchicalEnv


class ScenarioWrapper(object):
    def __init__(
        self,  
        wrapped_env,
        map_yaml_file,
        resolution_factor,
        distance_threshold,
        start_xy,
        start_theta,
        goal_xy,
        use_subgoal_from_global_planner,
        subgoal_dist,
        scenarios
    ):

        if isinstance(wrapped_env, NavGymHierarchicalEnv):
            self._wrapped_env = wrapped_env._wrapped_env
            self._wrapped_high_env = wrapped_env
        elif isinstance(wrapped_env, NavGymEnv):
            self._wrapped_env = wrapped_env
            self._wrapped_high_env = wrapped_env
        else:
            raise NotImplementedError

        self._wrapped_env.human_has_legs_ratio = 0.
        self._wrapped_env._sample_map = self._sample_map

        self.observation_space = self._wrapped_high_env.observation_space
        self.action_space = self._wrapped_high_env.action_space
        self.time_step = self._wrapped_env.time_step

        self.map_yaml_file = map_yaml_file
        self.resolution_factor = resolution_factor
        self.distance_threshold = distance_threshold
        self.start_xy = np.array(start_xy)
        self.start_theta = start_theta
        self.goal_xy = np.array(goal_xy)
        self.use_subgoal_from_global_planner= use_subgoal_from_global_planner
        self.subgoal_dist = subgoal_dist
        self.scenarios = np.array(scenarios)
        self.num_humans = len(scenarios)

        map_folder = os.path.dirname(self.map_yaml_file)
        map_name = os.path.basename(self.map_yaml_file)[:-5]
        self.map2d = CMap2D(map_folder, map_name)
        map_data = self.map2d.occupancy().T
        map_data[map_data > 0] = 100
        self.map_data = map_data.astype('int8')

        self.debug = np.array([0, 0, 0, 0, 0])

    @property
    def set_low_level_agent(self):
        return self._wrapped_high_env.set_low_level_agent

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
    def prev_obs(self):
        return self._wrapped_env.prev_obs

    @property
    def num_scan_stack(self):
        return self._wrapped_env.num_scan_stack

    def _sample_map(self):
        self._wrapped_env.map_info = {
            'data': self.map_data,
            'origin': (0, 0), # self.map2d.origin_xy(),
            'resolution': self.map2d.resolution() * self.resolution_factor,
            'width': self.map2d.occupancy().shape[1],
            'height': self.map2d.occupancy().shape[0]
        }
        x_min = self._wrapped_env.map_info['origin'][0]
        x_max = self._wrapped_env.map_info['origin'][0] + self._wrapped_env.map_info['width'] * self._wrapped_env.map_info['resolution'] 
        y_min = self._wrapped_env.map_info['origin'][1]
        y_max = self._wrapped_env.map_info['origin'][1] + self._wrapped_env.map_info['height'] * self._wrapped_env.map_info['resolution'] 
        self._wrapped_env.border = [(x_min, x_max), (y_min, y_max)]

        # === define costmap ===
        # =========================================================
        # Define a costmap with a higher resolution to quickly generate start, goal, and path points
        new_resolution = 0.25 #* self.resolution_factor
        scale = self._wrapped_env.map_info['resolution'] / 0.25
        new_height = int(scale * self._wrapped_env.map_info['height'])
        new_width = int(scale * self._wrapped_env.map_info['width'])
        self._wrapped_env.cost_map_info = {
            'data':cv2.resize(
                self._wrapped_env.map_info['data'].astype(np.uint8),
                (new_height, new_width),
                interpolation=cv2.INTER_NEAREST
            ),
            'origin': self._wrapped_env.map_info['origin'],
            'resolution': new_resolution,
            'width': new_width,
            'height': new_height
        }
        # To avoid generate waypoint near obstacle ...
        kernel = np.ones((7, 7)) # adjacent neighbors in all four directions (up, down, left, and right) 0.25 * 3 = 0.75m 
        self._wrapped_env.cost_map_info['data'] = cv2.filter2D(
            self._wrapped_env.cost_map_info['data'].astype(np.uint8), -1, kernel
        ).astype(np.uint8)
        self._wrapped_env.cost_map_info['data'][self._wrapped_env.cost_map_info['data'] > 0] = 100
        # =========================================================
        # for slow but accurate lidar computations (e.g., collision threshold)
        self._wrapped_env.contours = self._wrapped_env._get_contours()
        # for fast but inaccurate lidar computations
        MAX_DIST_IJ = self._wrapped_env.map_info['data'].shape[0] * self._wrapped_env.map_info['data'].shape[1]
        pyomap = range_libc.PyOMap(
            np.ascontiguousarray(self._wrapped_env.map_info['data'] >= 0.1))
        self._wrapped_env.rmnogpu = range_libc.PyRayMarching(pyomap, MAX_DIST_IJ)

    def reset(self):
        self._wrapped_high_env.reset()

        self._wrapped_env.prev_obs = None
        self._wrapped_env.prev_obs_queue = deque(maxlen=self._wrapped_env.num_scan_stack - 1) 
        self._wrapped_env.prev_human_actions = np.zeros((self.num_humans, 2))
        self._wrapped_env.prev_humans_obs_queue = [ # human use 3 stack scan
            deque(maxlen=2) for _ in range(self.num_humans)
        ]

        self._wrapped_env.robot.px = self.start_xy[0]
        self._wrapped_env.robot.py = self.start_xy[1]
        self._wrapped_env.robot.theta = self.start_theta
        self._wrapped_env.robot.gx = self.goal_xy[0]
        self._wrapped_env.robot.gy = self.goal_xy[1]

        if self.use_subgoal_from_global_planner:
            self.plan = find_path(
                self.start_xy[0], self.start_xy[1], self.goal_xy[0], self.goal_xy[1],
                self._wrapped_env.cost_map_info
            )

        self._wrapped_env.humans = []
        for scenario in self.scenarios:
            start = scenario[0]
            goal = scenario[1]

            path = find_path(
                start[0], start[1], goal[0], goal[1],
                self._wrapped_env.cost_map_info
            )
            human_theta = np.random.uniform(0, 2 * np.pi)
            human = Human(
                start[0], start[1], human_theta,
                goal[0], goal[1], 
                self.time_step
            )
            human.v_pref = np.random.uniform(
                self._wrapped_env.human_v_pref_range[0], self._wrapped_env.human_v_pref_range[1]
            )
            human.has_legs = False # np.random.random() < self._wrapped_env.human_has_legs_ratio
            waypoints = path_to_waypoints(path, interval=2) 

            human.waypoints = waypoints
            self._wrapped_env.humans.append(human)

        # human legs informations
        self._wrapped_env.distances_travelled_in_base_frame = np.zeros((len(self._wrapped_env.humans), 3))

        # keep human obs
        for i, human in enumerate(self._wrapped_env.humans):
            other_agents = [self._wrapped_env.robot] + [h for h in self._wrapped_env.humans if h != human]
            human_obs = self._wrapped_env._convert_obs(
                human, other_agents, self._wrapped_env.prev_obs, self._wrapped_env.prev_action,
                add_scan_noise=False, lidar_legs=False
            )
            # stack
            human_obs = self._wrapped_env._stack_scan(human_obs, self._wrapped_env.prev_humans_obs_queue[i], 3, human.n_angles)
            self._wrapped_env.prev_humans_obs_queue[i].append(human_obs)

        obs = self._wrapped_env._convert_obs(
            self._wrapped_env.robot, self._wrapped_env.humans, self._wrapped_env.prev_obs, self._wrapped_env.prev_action,
            add_scan_noise=True, lidar_legs=True
        )
        # stack
        obs = self._wrapped_env._stack_scan(obs, self._wrapped_env.prev_obs_queue, self._wrapped_env.num_scan_stack, self._wrapped_env.robot.n_angles)

        self._wrapped_env.prev_obs = obs
        self._wrapped_env.prev_obs_queue.append(obs)

        if self.use_subgoal_from_global_planner:
            d = np.linalg.norm(
                obs['achieved_goal'] - self.plan,
                axis=-1
            )
            local_goal_idxs = np.where([d > self.subgoal_dist])[1]
            if len(local_goal_idxs) > 0:
                local_goal_idx = np.where([d > self.subgoal_dist])[1][0]
            else:
                local_goal_idx = len(self.plan) - 1
            self.local_goal = self.plan[local_goal_idx]
            # update plan
            self.plan = self.plan[local_goal_idx:] 

            # use sub goal
            obs['desired_goal'] = self.local_goal

        return obs

    def step(self, action):
        self.debug = action

        obs, reward, _, info = self._wrapped_high_env.step(action)

        if self.use_subgoal_from_global_planner:
            d = np.linalg.norm(
                obs['achieved_goal'] - self.plan,
                axis=-1
            )
            local_goal_idxs = np.where([d > self.subgoal_dist])[1]
            if len(local_goal_idxs) > 0:
                local_goal_idx = np.where([d > self.subgoal_dist])[1][0]
            else:
                local_goal_idx = len(self.plan) - 1
            self.local_goal = self.plan[local_goal_idx]
            # update plan
            self.plan = self.plan[local_goal_idx:] 

            # use sub goal
            obs['desired_goal'] = self.local_goal
        
        done = False
        if info['distance'] < self.distance_threshold:
            done = True
            info['is_success'] = 1.
        elif info['is_crash']:
            done = True

        human_poses = np.zeros((len(self._wrapped_env.humans), 2))
        human_goals = np.zeros((len(self._wrapped_env.humans), 2))
        human_thetas = np.zeros(len(self._wrapped_env.humans))
        human_vs = np.zeros((len(self._wrapped_env.humans), 2))

        for i in range(len(self._wrapped_env.humans)):
            human_poses[i, 0] = self._wrapped_env.humans[i].px
            human_poses[i, 1] = self._wrapped_env.humans[i].py
            human_goals[i, 0] = self._wrapped_env.humans[i].gx
            human_goals[i, 1] = self._wrapped_env.humans[i].gy
            human_thetas[i] = self._wrapped_env.humans[i].theta
            human_vs[i, 0] = self._wrapped_env.humans[i].vx
            human_vs[i, 1] = self._wrapped_env.humans[i].vy

        info['human_poses'] = human_poses 
        info['human_goals'] = human_goals
        info['human_thetas'] = human_thetas 
        info['human_vs'] = human_vs

        return obs, reward, done, info

    def render(self, mode='human'):
        HEIGHT, WIDTH = 800, 800 # a good size to view

        img = self._wrapped_env.map_info['data'].copy().astype(np.float32)
        img[img == 0] = 1 # free space -> white
        img[img == 100] = 0 # obstacle -> black
        img = cv2.merge([img, img, img])

        # goal
        i,j = xy_to_ij([self._wrapped_env.robot.gx, self._wrapped_env.robot.gy], self._wrapped_env.map_info)
        r = xy_to_ij([1, 0], self._wrapped_env.map_info)[0]
        img = cv2.rectangle(
            img,
            (i - r, j - r),
            (i + r, j + r),
            (0, 0, 1),
            -1 # not fill
        )

        if self.use_subgoal_from_global_planner:
            # robot local goal from plan
            r = xy_to_ij([0.2, 0], self._wrapped_env.map_info)[0]
            i,j = xy_to_ij(np.array([self.local_goal[0], self.local_goal[1]]), self._wrapped_env.map_info)
            img = cv2.rectangle(
                img,
                (i - r, j - r),
                (i + r, j + r),
                (0, 0, 1),
                -1 # not fill
            )

        # humans local goal
        for human in self._wrapped_env.humans:  
            r = xy_to_ij([0.2, 0], self._wrapped_env.map_info)[0]
            i,j = xy_to_ij(np.array([human.gx, human.gy]), self._wrapped_env.map_info)
            img = cv2.rectangle(
                img,
                (i - r, j - r),
                (i + r, j + r),
                (1, 1, 0),
                -1 # not fill
            )

        # draw humans
        for human in self._wrapped_env.humans:
            i,j = xy_to_ij([human.px, human.py], self._wrapped_env.map_info)
            di, dj = xy_to_ij(
                [0.6 * np.cos(human.theta), 0.6 * np.sin(human.theta)], 
                self._wrapped_env.map_info,
                clip_if_outside=False
            )
            img = cv2.arrowedLine(
                img, (i, j), (i + di, j + dj), 
                (0, 0, 0), # color
                xy_to_ij([0.2, 0], self._wrapped_env.map_info)[0], # thickness (0.2m)
            )

            transformed_footprint = transform_xys(
                translation_matrix_from_xyz(human.px, human.py, 0),
                quaternion_matrix_from_yaw(human.theta),
                np.concatenate([human.footprint, [human.footprint[0]]])
            )
            for i in range(len(transformed_footprint) - 1):
                p1 = transformed_footprint[i]
                p2 = transformed_footprint[i + 1]
                p1_ij = xy_to_ij(p1, self._wrapped_env.map_info)
                p2_ij = xy_to_ij(p2, self._wrapped_env.map_info)
                cv2.line(
                    img,
                    tuple(p1_ij),
                    tuple(p2_ij),
                    (0, 0, 0), # color
                    1
                )

        # draw robot arrow
        i,j = xy_to_ij([self._wrapped_env.robot.px, self._wrapped_env.robot.py], self._wrapped_env.map_info)
        di, dj = xy_to_ij(
            [0.8 * np.cos(self._wrapped_env.robot.theta), 0.8 * np.sin(self._wrapped_env.robot.theta)], 
            self._wrapped_env.map_info,
            clip_if_outside=False
        )
        img = cv2.arrowedLine(
            img, (i, j), (i + di, j + dj), 
            (0, 0, 0), # color
            xy_to_ij([0.2, 0], self._wrapped_env.map_info)[0], # thickness (0.2m)
        )            

        # draw footprint
        transformed_footprint = transform_xys(
            translation_matrix_from_xyz(self._wrapped_env.robot.px, self._wrapped_env.robot.py, 0),
            quaternion_matrix_from_yaw(self._wrapped_env.robot.theta),
            np.concatenate([self._wrapped_env.robot.footprint, [self._wrapped_env.robot.footprint[0]]])
        )
        for i in range(len(transformed_footprint) - 1):
            p1 = transformed_footprint[i]
            p2 = transformed_footprint[i + 1]
            p1_ij = xy_to_ij(p1, self._wrapped_env.map_info)
            p2_ij = xy_to_ij(p2, self._wrapped_env.map_info)
            cv2.line(
                img,
                tuple(p1_ij),
                tuple(p2_ij),
                (0, 0, 0), # color
                1
            )
        # threshold footprint (collision check)
        transformed_threshold_footprint = transform_xys(
            translation_matrix_from_xyz(self._wrapped_env.robot.px, self._wrapped_env.robot.py, 0),
            quaternion_matrix_from_yaw(self._wrapped_env.robot.theta),
            np.concatenate([self._wrapped_env.robot.threshold_footprint, [self._wrapped_env.robot.threshold_footprint[0]]])
        )
        for i in range(len(transformed_threshold_footprint) - 1):
            p1 = transformed_threshold_footprint[i]
            p2 = transformed_threshold_footprint[i + 1]
            p1_ij = xy_to_ij(p1, self._wrapped_env.map_info)
            p2_ij = xy_to_ij(p2, self._wrapped_env.map_info)
            cv2.line(
                img,
                tuple(p1_ij),
                tuple(p2_ij),
                (0, 0, 0), # color
                1
            )
        # discomfort threshold footprint (discomfort check)
        transformed_discomfort_threshold_footprint = transform_xys(
            translation_matrix_from_xyz(self._wrapped_env.robot.px, self._wrapped_env.robot.py, 0),
            quaternion_matrix_from_yaw(self._wrapped_env.robot.theta),
            np.concatenate([self._wrapped_env.robot.discomfort_threshold_footprint, [self._wrapped_env.robot.discomfort_threshold_footprint[0]]])
        )
        for i in range(len(transformed_discomfort_threshold_footprint) - 1):
            p1 = transformed_discomfort_threshold_footprint[i]
            p2 = transformed_discomfort_threshold_footprint[i + 1]
            p1_ij = xy_to_ij(p1, self._wrapped_env.map_info)
            p2_ij = xy_to_ij(p2, self._wrapped_env.map_info)
            cv2.line(
                img,
                tuple(p1_ij),
                tuple(p2_ij),
                (0, 0, 0), # color
                1
            )

        # lidar
        observation_dict = observation_to_dict(
            self._wrapped_env.prev_obs['observation'],
            num_scan_stack=self._wrapped_env.num_scan_stack,
            n_angles=self._wrapped_env.robot.n_angles
        )
        scan = observation_dict['scan']
        theta = observation_dict['yaw']

        angles = np.linspace(self._wrapped_env.robot.angle_min,
                                self._wrapped_env.robot.angle_max - self._wrapped_env.robot.angle_increment,
                                self._wrapped_env.robot.n_angles) + theta

        lidar_points_x = self._wrapped_env.robot.px + scan * np.cos(angles)
        lidar_points_y = self._wrapped_env.robot.py + scan * np.sin(angles)
        lidar_points = np.hstack([
            lidar_points_x[:, None],
            lidar_points_y[:, None]
        ])  
        lidar_ijs = batch_xy_to_ij(lidar_points, self._wrapped_env.map_info)
        for ij in lidar_ijs[scan != self._wrapped_env.robot.range_max]: # do not plot max distance range
            i,j = ij
            img = cv2.circle(
                img, 
                (i, j),
                xy_to_ij([0.2, 0], self._wrapped_env.map_info)[0], # i,j space 상에서 radius
                (0., 1., 0.), # BGR color
                -1 # -1 means fill the circle
            )

        # # debug (human[0] lidar)
        # human_0 = self._wrapped_env.humans[0]
        # observation_dict = observation_to_dict(
        #     self._wrapped_env.prev_humans_obs_queue[0][-1]['observation'],
        #     num_scan_stack=3, # human use 3 stack rule
        #     n_angles=human_0.n_angles
        # )
        # scan = observation_dict['scan']
        # theta = observation_dict['yaw']

        # angles = np.linspace(human_0.angle_min,
        #                         human_0.angle_max - human_0.angle_increment,
        #                         human_0.n_angles) + theta

        # lidar_points_x = human_0.px + scan * np.cos(angles)
        # lidar_points_y = human_0.py + scan * np.sin(angles)
        # lidar_points = np.hstack([
        #     lidar_points_x[:, None],
        #     lidar_points_y[:, None]
        # ])  
        # lidar_ijs = batch_xy_to_ij(lidar_points, self._wrapped_env.map_info)
        # for ij in lidar_ijs[scan != human_0.range_max]: # do not plot max distance range
        #     i,j = ij
        #     img = cv2.circle(
        #         img, 
        #         (i, j),
        #         xy_to_ij([0.2, 0], self._wrapped_env.map_info)[0], # i,j space 상에서 radius
        #         (0., 0., 1.), # BGR color
        #         -1 # -1 means fill the circle
        #     )

        """
        Important: Except for the text, it should be drawn above this (flipped and rescaled from below)
        """
        # Flip it vertically to draw it in the world coordinate system
        img = np.flipud(img).copy()

        # resize for ease of viewing
        img = cv2.resize(img, (WIDTH, HEIGHT))

        # debug text
        for i, txt in enumerate(self._wrapped_env.render_obs_txt.split('\n') + self._wrapped_env.render_reward_txt.split('\n')):
            img = cv2.putText(
                img, 
                txt, 
                (50, 50 + i * 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, # fontscale
                (0, 0, 1), 
                2, # thickness
                cv2.LINE_AA
            ) 

        cv2.imshow("NavGym Env", img)

        # if isinstance(self._wrapped_high_env, NavGymHierarchicalEnv):
        #     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        #     [ax.bar(i, self.debug[i]) for i in range(5)]
        #     ax.set_xticklabels(['', 'crash', 'progress', 'forward', 'angular', 'discomfort'])
        #     ax.set_ylim([0, 0.2])
        #     fig.canvas.draw()
        #     img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        #     # Clean up
        #     plt.close(fig)
        #     cv2.imshow('high debug', img)

        cv2.waitKey(1)

def find_path(px, py, gx, gy, map_info):  
    grid = np.zeros_like(map_info['data'].T, dtype=np.float32)
    grid[map_info['data'].T == 100] = np.inf
    grid[map_info['data'].T == 0] = 255 

    start_ij = xy_to_ij([px, py], map_info)
    goal_ij = xy_to_ij([gx, gy], map_info)
    path = pyastar2d.astar_path(
        grid, start_ij, goal_ij, allow_diagonal=False)
    if path is not None:
        path = batch_ij_to_xy(path, map_info)
    return path


