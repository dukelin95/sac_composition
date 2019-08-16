"""
This file implements a wrapper for controlling the robot through end effector
movements instead of joint velocities. This is useful in learning pipelines
that want to output actions in end effector space instead of joint space.
"""

import os
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import Wrapper
from robosuite.controllers import SawyerIKController
from gym import spaces

class IKWrapper(Wrapper):
    env = None

    def __init__(self, env, action_repeat=1, gripper=False):
        """
        Initializes the inverse kinematics wrapper.
        This wrapper allows for controlling the robot through end effector
        movements instead of joint velocities.

        Args:
            env (MujocoEnv instance): The environment to wrap.
            action_repeat (int): Determines the number of times low-level joint
                control actions will be commanded per high-level end effector
                action. Higher values will allow for more precise control of
                the end effector to the commanded targets.
        """
        super().__init__(env)

        if self.env.mujoco_robot.name == "sawyer":

            self.controller = SawyerIKController(
                bullet_data_path=os.path.join(robosuite.models.assets_root, "bullet_data"),
                robot_jpos_getter=self._robot_jpos_getter,
            )
        else:
            raise Exception(
                "Only Sawyer robot environments supported for IK "
                "control currently."
            )

        self.action_repeat = action_repeat
        self.gripper = gripper
        if self.gripper:
            low = -np.array([.1, .1, .1, 1])
            high = np.array([.1, .1, .1, 1])

            self.action_space = spaces.Box(low=low, high=high)

        else:
            low = -np.array([.1, .1, .1])
            high = np.array([.1, .1, .1])

            self.action_space = spaces.Box(low=low, high=high)

        self.action_spec = low, high

    def set_robot_joint_positions(self, positions):
        """
        Overrides the function to set the joint positions directly, since we need to notify
        the IK controller of the change.
        """
        self.env.set_robot_joint_positions(positions)
        self.controller.sync_state()

    def _robot_jpos_getter(self):
        """
        Helper function to pass to the ik controller for access to the
        current robot joint positions.
        """
        return np.array(self.env._joint_positions)

    def reset(self):
        ret = super().reset()
        self.controller.sync_state()
        return ret

    def step(self, action):
        """
        Move the end effector(s) according to the input control.

        Args:
            action (numpy array): The array should have the corresponding elements.
                0-2: The desired change in end effector position in x, y, and z.
                (now constant orientation and gripper)
                previously:
                3-6: The desired change in orientation, expressed as a (x, y, z, w) quaternion.
                    Note that this quaternion encodes a relative rotation with respect to the
                    current gripper orientation. If the current rotation is r, this corresponds
                    to a quaternion d such that r * d will be the new rotation.
                *: Controls for gripper actuation.

        """
        if self.gripper:
            gripper_action = action[3:]
        else:
            gripper_action = np.ones(1) 
        constant_quat = np.array([0, 0, 0, 1])
        input_1 = self._make_input(np.concatenate((action[:3], constant_quat)), self.env._right_hand_quat)
        if self.env.mujoco_robot.name == "sawyer":
            velocities = self.controller.get_control(**input_1)
            low_action = np.concatenate([velocities, gripper_action])
        else:
            raise Exception(
                "Only Sawyer robot environments supported for IK "
                "control currently."
            )

        # keep trying to reach the target in a closed-loop
        for i in range(self.action_repeat):
            ret = self.env.step(low_action)
            if i + 1 < self.action_repeat:
                velocities = self.controller.get_control()
                if self.env.mujoco_robot.name == "sawyer":
                    low_action = np.concatenate([velocities, action[7:]])
                else:
                    raise Exception(
                        "Only Sawyer robot environments supported for IK "
                        "control currently."
                    )

        return ret

    def _make_input(self, action, old_quat):
        """
        Helper function that returns a dictionary with keys dpos, rotation from a raw input
        array. The first three elements are taken to be displacement in position, and a
        quaternion indicating the change in rotation with respect to @old_quat.
        """
        return {
            "dpos": action[:3],
            # IK controller takes an absolute orientation in robot base frame
            "rotation": T.quat2mat(T.quat_multiply(old_quat, action[3:7])),
        }
