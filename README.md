# PPO with World Models for gym-metacar environment

A combination of proximal policy optimization (PPO) and World Models is used to train a self-driving car in gym-metacar environment. More about gym-metacar can be found here: https://github.com/AI-Guru/gym-metacar

# Prerequisites

You should have chromedriver installed. 

### Mac

```python
brew cask install google-chrome
brew cask install chromedriver
```

### Linux

```python
apt install chromium-chromedriver
apt-get install -y libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1
```

### Environment specifications

You can skip this step totally. It only describes how to change the resolution of the lidar, so you could use the pretrained models and run the scripts without any changes. The default lidar resolution is 5x5.

I used a lidar resolution of 32x32, so to change the default one you would need to change the specification of the environment. The script with the environment should be in the directory of your environment:

```python
path_to_environment/lib/python3.6/site-packages/gym_metacar/envs
```

Then you should insert these 2 lines to the appropriate positions in the section for observation space.

```python
lidar_space = spaces.Box(low=-1, high=1, shape=(32, 32), dtype=np.float32)
linear_space = spaces.Box(low=-1, high=1, shape=(32 * 32 + 1,), dtype=np.float32)
```

And finally you should add this line to the second if/else clause in the trigger environment initialization section:

```python
script += 'env.setAgentLidar({pts: 32, width: 5, height: 5, pos: -1.5});' + "\n"
```
 

