from rlkit.envs.wrappers.discretize_env import DiscretizeEnv
from rlkit.envs.wrappers.history_env import HistoryEnv
try:
    from rlkit.envs.wrappers.image_mujoco_env import ImageMujocoEnv
    from rlkit.envs.wrappers.image_mujoco_env_with_obs import ImageMujocoWithObsEnv
except:
    print('rlkit/envs/wrappers/__init__.py: can not import ImageMujocoEnv. Mujoco not installed')
from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv
from rlkit.envs.proxy_env import ProxyEnv
from rlkit.envs.wrappers.reward_wrapper_env import RewardWrapperEnv
from rlkit.envs.wrappers.stack_observation_env import StackObservationEnv


__all__ = [
    'DiscretizeEnv',
    'HistoryEnv',
    'ImageMujocoEnv',
    'ImageMujocoWithObsEnv',
    'NormalizedBoxEnv',
    'ProxyEnv',
    'RewardWrapperEnv',
    'StackObservationEnv',
]