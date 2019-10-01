from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.sawyer import SawyerEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import TableTopTask, UniformRandomSampler, ObjectPositionSampler

from robosuite.controllers import SawyerIKController
import robosuite
import pybullet as p
import os
from os.path import join as pjoin
from gym import spaces

class SawyerReach(SawyerEnv):
    """
    This class corresponds to the a primitive policy for the reach task on the Sawyer robot arm.

    Uniform random sample on x, y, or z axis in range of 20 to 60 cm for x and y, 0 to 40 cm for z (center of table)
    """

    def __init__(
        self,
        gripper_type="TwoFingerGripper",
        lower_goal_range = [-0.1, -0.1, -0.1],
        upper_goal_range = [0.1, 0.1, 0.2],
        seed=False,
        random_arm_init = None, # please pass in [low, high]
        table_full_size=(0.8, 0.8, 0.6),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        placement_initializer=None,
        gripper_visualization=False,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
    ):
        """
        Args:
            prim_axis (str): which axis is being explored

            gripper_type (str): type of gripper, used to instantiate￼
                gripper models from gripper factory.

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """
        self.upper_goal_range = upper_goal_range
        self.lower_goal_range = lower_goal_range
        self.random_arm_init = random_arm_init
        self.seed = seed

        self.distance_threshold = 0.015
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        # object placement initializer
        self.placement_initializer = placement_initializer

        # order: di["eef_pos"] di["gripper_qpos"] di["gripper_to_marker"] self.goal
        # dim: 3, 2, 3, 3
        # low = -np.ones(11) * np.inf
        # high = np.ones(11) * np.inf
        # self.observation_space = spaces.Box(low=low, high=high)

        self.controller = SawyerIKController(
                bullet_data_path=os.path.join(robosuite.models.assets_root, "bullet_data"),
                robot_jpos_getter=self._robot_jpos_getter,
            )
        super().__init__(
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        self.mujoco_objects = OrderedDict([])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        # self.cube_body_id = self.sim.model.body_name2id("cube")
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]
        # self.cube_geom_id = self.sim.model.geom_name2id("cube")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # reset positions of objects
        self.model.place_objects()

        if self.random_arm_init is not None:
            if self.seed:
                constant_quat = np.array([-0.01704371, -0.99972409, 0.00199679, -0.01603944])
                target_position = np.array([0.5 + np.random.uniform(0.0, 0.0),
                                            np.random.uniform(0.0 ,0.0),
                                            # self.table_full_size[2] + 0.15211762])
                                            0.8 + 0.20211762])
            else:
            # random initialization of arm
                constant_quat = np.array([-0.01704371, -0.99972409, 0.00199679, -0.01603944])
                target_position = np.array([0.5 + np.random.uniform(self.random_arm_init[0], self.random_arm_init[1]),
                                            np.random.uniform(self.random_arm_init[0], self.random_arm_init[1]),
                                            self.table_full_size[2] + 0.20211762 ])
            self.controller.sync_ik_robot(self._robot_jpos_getter(), simulate=True)
            joint_list = self.controller.inverse_kinematics(target_position, constant_quat)
            init_pos = np.array(joint_list)


        else:
            # default robosuite init
            init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
            init_pos += np.random.randn(init_pos.shape[0]) * 0.02

        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

    def reset(self):
        self._destroy_viewer()
        self._reset_internal()
        self.sim.forward()
        
        # reset goal (marker)
        gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
        distance = np.random.uniform(self.lower_goal_range, self.upper_goal_range)
        while np.linalg.norm(distance) <= (self.distance_threshold * 1.75):
            distance = np.random.uniform(self.lower_goal_range, self.upper_goal_range)
        if self.seed:
            distance = np.array((0.05, -0.1, 0.1))

        self.goal = gripper_site_pos + distance

        return self._get_observation()

    def _robot_jpos_getter(self):
        return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])

    def reward(self, action=None):
        """
        Reward function for the task.

        The dense reward has one component.

            Reaching: in [0, 1], to encourage the arm to reach the marker

        The sparse reward only consists of the lifting component.

        Args:
            action (np array): unused for this task

        Returns:
            reward (float): the reward
        """
     #   velocity_pen = np.linalg(np.array(
     #       [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
     #  ))
        marker_pos = self.goal
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]

        return self.compute_reward(gripper_site_pos, marker_pos, None)

    # for goalenv wrapper
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        d = np.linalg.norm(achieved_goal - desired_goal)
        if self.reward_shaping: # dense
            reward = 1 - np.tanh(10 * d)
            if d <= self.distance_threshold:
                reward = 10.0
        else: # sparse (-1 or 0)
            #reward = -np.float32(d > distance_threshold)
            if d > self.distance_threshold:
                reward = -1.0
            else:
                reward = 60.0

        return reward

    # for goalenv wrapper
    # TODO check the desired_goal/achieved_goal for indexing
    def get_goalenv_dict(self, obs_dict):
        # using only object-state and robot-state
        ob_lst = []
        di = {}
        for key in obs_dict:
            if key in ["robot-state", "object-state"]:
                ob_lst.append(obs_dict[key])

        di['observation'] = np.concatenate(ob_lst)
        di['desired_goal'] = obs_dict['object-state'][0:3]
        di['achieved_goal'] = obs_dict['robot-state'][23:26]

        return di

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
        # camera observations
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
            else:
                di["image"] = camera_obs

        # low-level object information
        if self.use_object_obs:
            gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di["gripper_to_marker"] = gripper_site_pos - self.goal

            di["object-state"] = np.concatenate(
                [di["gripper_to_marker"], self.goal]
            )

        return di

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to cube
        if self.gripper_visualization:
            # get distance to cube
            cube_site_id = self.sim.model.site_name2id("cube")
            dist = np.sum(
                np.square(
                    self.sim.data.site_xpos[cube_site_id]
                    - self.sim.data.get_site_xpos("grip_site")
                )
            )

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba

