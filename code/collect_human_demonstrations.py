"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import os
import shutil
import time
import argparse
import datetime
import h5py
from glob import glob
import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper
from robosuite.wrappers import DataCollectionWrapper
from robosuite.models.tasks import UniformRandomSampler

import dateutil.tz

def collect_human_trajectory(env, device):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env: environment to control
        device (instance of Device class): to receive controls from the device
    """

    obs = env.reset()

    # rotate the gripper so we can see it easily
    # env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

    env.viewer.set_camera(camera_id=2)
    env.render()

    is_first = False

    # episode terminates on a spacenav reset input or if task is completed
    reset = False
    task_completion_hold_count = -1 # counter to collect 10 timesteps after reaching goal
    device.start_control()
    count = 0
    while not reset:
        count += 1
        state = device.get_controller_state()
        dpos, rotation, grasp, reset = (
            state["dpos"],
            state["rotation"],
            state["grasp"],
            state["reset"],
        )

        # convert into a suitable end effector action for the environment
        current = env._right_hand_orn
        drotation = current.T.dot(rotation)  # relative rotation of desired from current
        dquat = T.mat2quat(drotation)
        grasp = grasp - 1.  # map 0 to -1 (open) and 1 to 0 (closed halfway)
        action = np.concatenate([dpos, dquat, [grasp]])

        obs, reward, done, info = env.step(action)
        if count%50 == 0: print(env.env._joint_positions)

        if is_first:
            is_first = False

            # We grab the initial model xml and state and reload from those so that
            # we can support deterministic playback of actions from our demonstrations.
            # This is necessary due to rounding issues with the model xml and with
            # env.sim.forward(). We also have to do this after the first action is 
            # applied because the data collector wrapper only starts recording
            # after the first action has been played.
            initial_mjstate = env.sim.get_state().flatten()


            xml_str = env.model.get_xml()
            env.reset_from_xml_string(xml_str)
            env.sim.reset()
            env.sim.set_state_from_flattened(initial_mjstate)
            env.sim.forward()
            env.viewer.set_camera(camera_id=2)

        env.render()
        env.env.viewer.viewer.add_marker(pos=env.env.goal,
                                         size=np.array((0.02, 0.02, 0.02)), label='goal',
                                         rgba=[1, 0, 0, 0.5])

        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1 # latched state, decrement count
            else:
                task_completion_hold_count = 10 # reset count on first success timestep
        else:
            task_completion_hold_count = -1 # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file, and another directory that contains the 
    raw model.xml files.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - name of corresponding model xml in `models` directory
            states (dataset) - flattened mujoco states
            joint_velocities (dataset) - joint velocities applied during demonstration
            gripper_actuations (dataset) - gripper controls applied during demonstration
            right_dpos (dataset) - end effector delta position command for
                single arm robot or right arm
            right_dquat (dataset) - end effector delta rotation command for
                single arm robot or right arm
            left_dpos (dataset) - end effector delta position command for
                left arm (bimanual robot only)
            left_dquat (dataset) - end effector delta rotation command for
                left arm (bimanual robot only)

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file and model xmls. 
            The model xmls will be stored in a subdirectory called `models`.
    """

    # store model xmls in this directory
    model_dir = os.path.join(out_dir, "models")
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        joint_velocities = []
        gripper_actuations = []
        right_dpos = []
        right_dquat = []
        left_dpos = []
        left_dquat = []

        # for demonstrations
        action = []
        reward = []
        observation = []
        terminal = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                joint_velocities.append(ai["joint_velocities"])
                gripper_actuations.append(ai["gripper_actuation"])
                right_dpos.append(ai.get("right_dpos", []))
                right_dquat.append(ai.get("right_dquat", []))
                left_dpos.append(ai.get("left_dpos", []))
                left_dquat.append(ai.get("left_dquat", []))

                action.append(ai["action"])
                reward.append(ai["reward"])
                observation.append(ai["observation"])
                terminal.append(ai["terminal"])
        if len(states) == 0:
            continue

        # Delete the first actions and the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action.
        del states[-1]
        del joint_velocities[0]
        del gripper_actuations[0]
        del right_dpos[0]
        del right_dquat[0]
        del left_dpos[0]
        del left_dquat[0]

        del action[0]
        del reward[-1]
        del observation[-1]
        del terminal[-1]

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model file name as an attribute
        ep_data_grp.attrs["model_file"] = "model_{}.xml".format(num_eps)

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("joint_velocities", data=np.array(joint_velocities))
        ep_data_grp.create_dataset(
            "gripper_actuations", data=np.array(gripper_actuations)
        )
        ep_data_grp.create_dataset("right_dpos", data=np.array(right_dpos))
        ep_data_grp.create_dataset("right_dquat", data=np.array(right_dquat))
        ep_data_grp.create_dataset("left_dpos", data=np.array(left_dpos))
        ep_data_grp.create_dataset("left_dquat", data=np.array(left_dquat))

        ep_data_grp.create_dataset("action", data=np.array(action))
        ep_data_grp.create_dataset("reward", data=np.array(reward))
        ep_data_grp.create_dataset("observation", data=np.array(observation))
        ep_data_grp.create_dataset("terminal", data=np.array(terminal))

        # copy over and rename model xml
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        shutil.copy(xml_path, model_dir)
        os.rename(
            os.path.join(model_dir, "model.xml"),
            os.path.join(model_dir, "model_{}.xml".format(num_eps)),
        )

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = robosuite.__version__
    grp.attrs["env"] = env_name

    f.close()

def timestamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f-%Z')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robosuite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="SawyerLift")
    parser.add_argument("--device", type=str, default="keyboard")
    args = parser.parse_args()

    # create original environment
    # env = robosuite.make(
    #     args.environment,
    #     ignore_done=True,
    #     use_camera_obs=False,
    #     has_renderer=True,
    #     control_freq=100,
    #     gripper_visualization=True,
    # )
    from robosuite.environments.sawyer_test_reachpick import SawyerReachPick
    from robosuite.environments.sawyer_test_pick import SawyerPrimitivePick
    random_arm_init = [-0.0, 0.0]
    # random_arm_init = None
    instructive = 1.0
    decay = 3e-6
    env = SawyerPrimitivePick(
        instructive=instructive,
        decay=decay,
        random_arm_init=random_arm_init,
        has_renderer=True,
        reward_shaping=False,
        horizon=150,
        ignore_done =True,
        placement_initializer=UniformRandomSampler(
            x_range=[-0.01, 0.01],
            y_range=[-0.01, 0.01],
            ensure_object_boundary_in_range=False,
            z_rotation=None,
        ),

        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=100, )
    # reward_shaping = False
    # horizon = 500
    # env = SawyerReachPick(
    #     # playable params
    #     random_arm_init=random_arm_init,
    #     has_renderer=True,
    #     reward_shaping=reward_shaping,
    #     horizon=horizon,
    #     placement_initializer=UniformRandomSampler(
    #         x_range=[-0.01, 0.01],
    #         y_range=[-0.01, 0.01],
    #         ensure_object_boundary_in_range=False,
    #         z_rotation=None,
    #     ),
    #
    #     # constant params
    #     has_offscreen_renderer=False,
    #     ignore_done=True,
    #     use_camera_obs=False,
    #     use_object_obs=True,
    #     control_freq=100,
    #     gripper_visualization=True,)

    # enable controlling the end effector directly instead of using joint velocities
    env = IKWrapper(env)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard()
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse()
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    # new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    new_dir = os.path.join(args.directory, timestamp())
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device)
        print("next")
        gather_demonstrations_as_hdf5(tmp_directory, new_dir)
