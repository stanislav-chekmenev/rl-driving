import gym
try:
    import gym_metacar
except:
    import sys
    sys.path.append(".")
    sys.path.append("..")
    import gym_metacar

from gym_metacar.wrappers import *
from baselines.common.vec_env import *
from rldriving.drive.wm_ppo_policy import WMPPODriver
from rldriving.envs.world_model_wrapper import WorldModelWrapper


# Create environment
nstack = 12
env_id = "metacar-level3-continuous-v0"
env = gym.make(env_id)
env.enable_webrenderer()
env = StepLimitTerminateWrapper(env, 2048)
env = ClipRewardsWrapper(env)
env = WorldModelWrapper(env)
env = DummyVecEnv([lambda:env])
env = VecFrameStack(env, nstack=nstack)

# Get input_shape
input_shape = nstack * 269

# Create policy
driver = WMPPODriver(input_shape)
driver.load(filepath="control/models/", name='wm_ppo_trained')

# Drive around the city
for episode in range(1):
    print("Episode ", episode)
    obs = env.reset()
    done = False
    while done == False:
        action = driver(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

env.close()
