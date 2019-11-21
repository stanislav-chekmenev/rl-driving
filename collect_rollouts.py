import gym
import h5py
import ray
import time


try:
    import gym_metacar
except:
    import sys
    sys.path.append(".")
    sys.path.append("..")
    import gym_metacar
from gym_metacar.wrappers import *


@ray.remote
def rollout(i, step_limit, num_episodes, lidar_shape):
    epsilon = 0.5
    print("Creating environment ", i)
    env = gym.make("metacar-level3-continuous-v0")
    env = LidarObservationWrapper(env)
    env = ClipRewardsWrapper(env)
    env = StepLimitTerminateWrapper(env, step_limit)
    
    print('Creating H5 files for observations and actions for environment ', i)
    # Create a file to write observations of the lidar into
    with h5py.File('env_' + str(i) + '_observations.h5', 'w', libver='latest') as fo:
        fo.create_dataset('lidar', data=np.zeros((lidar_shape, lidar_shape)), compression="gzip", chunks=True, maxshape=(None, 128))
    
    # Create a file to write actions into
    with h5py.File('env_' + str(i) + '_actions.h5', 'w', libver='latest') as fa:
        fa.create_dataset('actions', data=np.array([0, 0]).reshape(1, 2), compression="gzip", chunks=True, maxshape=(None, 2))

    for episode in range(num_episodes):
        print('Running episode {} in environment {}'.format(episode, i))
        env.reset()
        done = False
        step = 0
        observations = []
        actions = []
        if np.random.uniform(size=1) < epsilon:
            if step < np.floor(step_limit / 2):
                while done == False:
                    action = [1, env.action_space.sample()[1]]
                    obs, _, done, info = env.step(action)
                    step += 1
                    done = done
                    observations.append(obs)
                    actions.append(np.array(action).reshape(1, 2))
            else:
                while done == False:
                    action = [-1, env.action_space.sample()[1]]
                    obs, _, done, info = env.step(action)
                    step += 1
                    done = done
                    observations.append(obs)
                    actions.append(np.array(action).reshape(1, 2))
        else:
            if step < np.floor(step_limit / 2):
                while done == False:
                    action = [-1, env.action_space.sample()[1]]
                    obs, _, done, info = env.step(action)
                    step += 1
                    done = done
                    observations.append(obs)
                    actions.append(np.array(action).reshape(1, 2))
            else:
                while done == False:
                    action = [1, env.action_space.sample()[1]]
                    obs, _, done, info = env.step(action)
                    step += 1
                    done = done
                    observations.append(obs)
                    actions.append(np.array(action).reshape(1, 2))
        
        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        
        print('Appending observations and actions to files for environment {} and episode {}'.format(i, episode))
        with h5py.File('env_' + str(i) + '_observations.h5', 'a') as fo:
            fo["lidar"].resize((fo["lidar"].shape[0] + observations.shape[0]), axis = 0)
            fo["lidar"][-observations.shape[0]:] = observations

        with h5py.File('env_' + str(i) + '_actions.h5', 'a') as fa:
            fa["actions"].resize((fa["actions"].shape[0] + actions.shape[0]), axis = 0)
            fa["actions"][-actions.shape[0]:] = actions
    
    return None
                
    
def parallel_rollout(num_env, step_limit, num_episodes, lidar_shape):
# note: maybe run this twice to warmup the system
    ray.init()
    ray.get([rollout.remote(i, step_limit, num_episodes, lidar_shape) for i in range(num_env)])
    ray.shutdown()
    return None


def main():
    start = time.time()
    parallel_rollout(num_env=1, step_limit=1, num_episodes=1, lidar_shape=32)
    print('Rollouts have been finished')
    print('Elapsed time in seconds: {}'.format(time.time() - start))
    
if __name__ == "__main__":
    main()