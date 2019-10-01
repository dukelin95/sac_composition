# import numpy as np
# from robosuite.environments.sawyer_test_reach import SawyerReach

# from crl_env_wrapper import CRLWrapper

random_arm_init = [-0.1, 0.1]
render = False
reward_shaping = False
horizon = 100
# env = CRLWrapper(SawyerReach(
#     # playable params
#     random_arm_init=random_arm_init,
#     has_renderer=render,
#     reward_shaping=reward_shaping,
#     horizon=horizon,
#
#     # constant params
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
#     use_object_obs=True,
#     control_freq=100,
# ))
