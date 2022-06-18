---
title: 'Training a deep reinforcement learning agent to play Super Mario Bros'
tags: [Reinforcement Learning, AI, Super Mario Bros, DDQN, PPO, A2C]
layout: post
mathjax: true
categories: [Reinforcement Learning]
permalink: /blog/:title/
---
{% assign counter = 1 %}
{% assign counter2 = 1 %}
{% assign link = "https://raw.githubusercontent.com/tomchiang30115/tomchiang30115.github.io/main/_posts/" %}
{% assign date = page.date | date: "%Y-%m-%d" %}
{% assign filename = page.title | remove: " -" | replace: " ", "-" | downcase %}

<br>
[![jpeg]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.jpeg#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.jpeg)
{% assign counter = counter | plus: 1 %} 
<br>

This is a group project I did in reinforcement learning module, where I worked with 5 other members to create this deep reinforcement learning algorithm that plays the game Super Mario Bros by itself. The report is displayed below.

<br>
<video autoplay loop muted playsinline>
<source src="https://tomchiang30115.github.io/_post_/2022-06-18-DDQN-mario/2.webm#center" type="video/webm", style="text-align:center;">
<source src="https://tomchiang30115.github.io/_post_/2022-06-18-DDQN-mario/2.mp4#center" type="video/mp4" style="text-align:center;"></video>
<br><center>The fully trained DDQN agent.</center>


<div id="adobe-dc-view" style="width: 100%;"></div>
<script src="https://documentcloud.adobe.com/view-sdk/main.js"></script>
<script type="text/javascript">
	document.addEventListener("adobe_dc_view_sdk.ready", function(){ 
		var adobeDCView = new AdobeDC.View({clientId: "3e66a9482764407187a81a0bf601400a", divId: "adobe-dc-view"});
		adobeDCView.previewFile({
			content:{location: {url: "https://tomchiang30115.github.io/pdf/RL_Project_Group_30_v2.pdf"}},
			metaData:{fileName: "RL_Project_Group_30_v2.pdf"}
		}, {embedMode: "IN_LINE"});
	});
</script>

### The DDQN trainning algorithm

```python

###########
# Imports #
###########

# Mario Envrionment Libraries
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Other Libraries
import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from matplotlib import pyplot as plt
from collections import deque
import torch.nn as nn
import torch
import time
import random
import copy
import numpy as np
import torchvision
import os
###############
# Agent Class #
###############

class Agent:
    def __init__(self, num_actions, replay_buffer_size=100000, num_replay_samples=32, model_path=None):
        self.num_actions = num_actions
        self.replay_buffer = deque(maxlen=replay_buffer_size) # Stores up to the given number of past experiences (called 'D' in slides)
        self.num_replay_samples = num_replay_samples
        self.epsilon = 1.0
        self.epsilon_decay = 0.99999975
        self.epsilon_min = 0.1
        self.gamma = 0.9
        self.loss = torch.nn.HuberLoss()
        self.learning_rate = 0.000025
        
        # Allow torch to use GPU for processing, if avaiable
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        print("Torch device: ", device)
        
        self.q1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        ).to(device)
        
        self.previous_episodes = 0
        self.previous_steps = 0
        
        self.optimizer = torch.optim.Adam(self.q1.parameters(), lr=self.learning_rate)
        
        # Check if previously trained model has been provided
        if model_path != None:
            loaded_model = torch.load(model_path)
            self.q1.load_state_dict(loaded_model['model_state_dict'])
            self.q1.train()
            self.optimizer = torch.optim.Adam(self.q1.parameters(), lr=loaded_model['learning_rate'])
            self.optimizer.load_state_dict(loaded_model['optimizer_state_dict'])
            self.previous_episodes = loaded_model['num_episodes']
            self.previous_steps = loaded_model['num_steps']
            self.epsilon = loaded_model['epsilon']
            
        self.q2 = copy.deepcopy(self.q1) # Create target action-value network as copy of action-value network q1
        
        # Prevent weights being updated in target network
        for param in self.q2.parameters():
            param.requires_grad = False
    
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
    
    def update_logs(self, save_location, log_interval, episode, step, average_episode_loss, 
                    average_episode_reward, average_episode_distance, flag_count, death_count, timeout_count):
        print("Updating logs")
        self.log_file = open(save_location + 'log.txt', 'a')
        self.log_file.write(str(episode+self.previous_episodes) + "," 
                            + str(step+self.previous_steps) + "," 
                            + str(average_episode_reward) + ","
                            + str(average_episode_distance) + ","
                            + str(average_episode_loss) + ","
                            + str(flag_count) + ","
                            + str(death_count) + ","
                            + str(timeout_count) + ","
                            + str(self.epsilon) + "\n")
        self.log_file.close()
    
    def save_model(self, step, episode, save_location):
        torch.save({
            'model_state_dict': self.q1.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'num_episodes': episode + self.previous_episodes,
            'num_steps': step + self.previous_steps,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate
        },
            save_location + str(step + self.previous_steps) + '.pth')
    
    def get_action(self, state):
        # epsilon-greedy policy
        if (random.uniform(0, 1) < self.epsilon):
            # Explore randomly
            random_action = random.randint(0, self.num_actions-1)
            #print("Selected action:", random_action)
            return random_action
        else:
            # Follow policy greedily (A_t = argmax_q1(S_t, a, theta1))
            state = torch.tensor(state.__array__()).squeeze().cuda() # Squeeze to get rid of uneccessary channel dimension
            #print(state.shape)
            state = torchvision.transforms.functional.convert_image_dtype(state, dtype=torch.float)
            state = state.unsqueeze(0) # Add first dimension to match shape of minibatches being fed into network
            #print(state.shape)
            
            action_values = self.q1(state)
            #print(action_values)
            best_action = torch.argmax(action_values, axis=1).item()
            #print("Selected action:", best_action)
            return best_action
        
    def add_to_replay_buffer(self, state, chosen_action, reward, next_state, done):
        # Convert to tensors
        state = torch.tensor(state.__array__()).squeeze().cuda()
        chosen_action = torch.tensor([chosen_action]).cuda()
        reward = torch.tensor([reward]).cuda()
        next_state = torch.tensor(next_state.__array__()).squeeze().cuda()
        done = torch.tensor([int(done)]).cuda()
        
        self.replay_buffer.append((state, chosen_action, reward, next_state, done))
        
    def sync_networks(self):
        self.q2.load_state_dict(self.q1.state_dict())
        
    def perform_updates(self):
        num_samples = self.num_replay_samples
        
        # Check if replay buffer has enough samples for full minibatch
        # Don't perform updates until enough samples in replay buffer
        if len(self.replay_buffer) < self.num_replay_samples:
            #print("Replay buffer only has", len(self.replay_buffer), "transitions. Skipping updates...")
            return
        
        # Randomly select a number of transitions from the replay buffer
        minibatch = random.sample(self.replay_buffer, self.num_replay_samples)
        state, chosen_action, reward, next_state, done = map(torch.stack, zip(*minibatch)) # https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html?highlight=transformer
        chosen_action = chosen_action.squeeze()
        reward = reward.squeeze()
        done = done.squeeze()
        #print("Shape of batch:", next_state.shape)
        
        # Get TD Value Estimate
        state = torchvision.transforms.functional.convert_image_dtype(state, dtype=torch.float)
        estimated_value = self.q1(state)[np.arange(0, self.num_replay_samples), chosen_action]
        
        # Get TD Value Target
        with torch.no_grad(): # Disable gradient calculation: https://pytorch.org/docs/stable/generated/torch.no_grad.html
            next_state = torchvision.transforms.functional.convert_image_dtype(next_state, dtype=torch.float)
            next_state_Q = self.q1(next_state)
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.q2(next_state)[np.arange(0, self.num_replay_samples), best_action]

        target_value = (reward + (1 - done.float()) * self.gamma * next_Q).float()
        # print("TD Target:", target_value)
        
        # Calculate loss using Huber Loss
        loss = self.loss(estimated_value, target_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Stuff for logging
        return loss.item()
##############################
# Training Environment Setup #
##############################

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
chosen_inputs = SIMPLE_MOVEMENT
env = JoypadSpace(env, chosen_inputs)

# Environment Preprocessing - Apply wrappers to the environment to reduce load on NN
num_skip_frames = 4
env = MaxAndSkipEnv(env, skip=num_skip_frames) # Skip the given number of frames
env = GrayScaleObservation(env) # Convert state from RGB to grayscale (1/3 number of pixels for NN to process)
env = ResizeObservation(env, shape=84) # Scale each frame to 64x64 pixels (15x fewer pixels for NN to process)
env = FrameStack(env, 4) # Stack the last 4 observations together to give NN temporal awareness
# Final observation shape: (        4         ,   84   ,   84    ,      1      )
#                           num_stacked_frames  height    width    num_channels
#######################
# Agent Training Loop #
#######################

resuming_training = False
saved_model_path = './models/20220418-175357/30000.pth'

if resuming_training:
    agent = Agent(len(chosen_inputs), model_path=saved_model_path) # Continue training existing model
else:
    agent = Agent(len(chosen_inputs)) # Train NEW agent, set resuming_training to false if training from scratch

num_episodes = 1000000
sync_interval = 10000 #(steps)
save_interval = 10000 #(steps)
log_interval = 20 #(episodes)
timestamp = time.strftime("%Y%m%d-%H%M%S")
save_location = './models/lr0.000025/' + timestamp + '/'
os.mkdir(save_location)

step = 1
rewards_per_episode = []
distance_per_episode = []
average_loss_per_episode = []
death_count = 0
timeout_count = 0
flag_count = 0
for episode in range(num_episodes):
    episode_loss = 0.0
    num_losses = 0
    average_episode_loss = 0.0
    episode_reward = 0.0
    num_episode_steps = 1
    state = env.reset()

    while True:
            
        # Choose and execute an action following epsilon-greedy policy
        chosen_action = agent.get_action(state)
        next_state, reward, done, info = env.step(chosen_action)
        episode_reward += reward
        
        # Add the transition to the replay buffer
        agent.add_to_replay_buffer(state, chosen_action, reward, next_state, done)
        
        # Update random sample of transitions from the replay buffer
        loss = agent.perform_updates()
        if loss is not None:
            episode_loss += loss
            num_losses += 1
            average_episode_loss = episode_loss / num_losses
        
        agent.decay_epsilon()
        
        if step % 10000 == 0:
            print("Done", step, "steps,", episode, "completed episodes")
        
        # Sync parameters of target network every so often
        if step % sync_interval == 0:
            agent.sync_networks()
            
        # Save model every so often
        if step % save_interval == 0:   
            agent.save_model(step, episode, save_location)
        
        #env.render()
        
        state = next_state
        step += 1 # Total steps over all episodes
        num_episode_steps += 1 # Steps in this episode
        
        if done:
            rewards_per_episode.append(episode_reward)
            distance_per_episode.append(info['x_pos'])
            average_loss_per_episode.append(average_episode_loss)
            
            if info['time'] == 0:
                timeout_count += 1
            elif info['flag_get']:
                flag_count += 1
            else:
                death_count += 1
            
            # Log episode stats
            if episode % log_interval == 0 and episode != 0:
                average_episode_reward = sum(rewards_per_episode) / len(rewards_per_episode)
                average_episode_distance = sum(distance_per_episode) / len(distance_per_episode)
                average_loss_across_episodes = sum(average_loss_per_episode) / len(average_loss_per_episode)
                agent.update_logs(save_location, log_interval, episode, step, average_loss_across_episodes, 
                                  average_episode_reward, average_episode_distance, flag_count, death_count, timeout_count)
                rewards_per_episode = []
                distance_per_episode = []
                average_loss_per_episode = []
                timeout_count = 0
                flag_count = 0
                death_count = 0
            
            break
env.close()


```

### The DDQN evaluation algorithm

```python

###########
# Imports #
###########

# Mario Envrionment Libraries
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Other Libraries
import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from matplotlib import pyplot as plt
from collections import deque
import torch.nn as nn
import torch
import time
import random
import copy
import numpy as np
import torchvision
import os
###############
# Agent Class #
###############

class Agent:
    def __init__(self, num_actions, model_path=None):
        self.num_actions = num_actions
        self.epsilon = 0.0
        
        # Allow torch to use GPU for processing, if avaiable
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        print("Torch device: ", device)
        
        self.q1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        self.q1.to(device)
        
        # Check if previously trained model has been provided
        if model_path != None:
            loaded_model = torch.load(model_path)
            self.q1.load_state_dict(loaded_model['model_state_dict'])
            self.q1.eval()
            self.epsilon = loaded_model['epsilon']
    
    def get_action(self, state):
        # epsilon-greedy policy
        if (random.uniform(0, 1) < self.epsilon):
            # Explore randomly
            random_action = random.randint(0, self.num_actions-1)
            #print("Selected action:", random_action)
            return random_action
        else:
            # Follow policy greedily (A_t = argmax_q1(S_t, a, theta1))
            state = torch.tensor(state.__array__()).squeeze().cuda() # Squeeze to get rid of uneccessary channel dimension
            #print(state.shape)
            state = torchvision.transforms.functional.convert_image_dtype(state, dtype=torch.float)
            state = state.unsqueeze(0) # Add first dimension to match shape of minibatches being fed into network
            #print(state.shape)
            
            action_values = self.q1(state)
            #print(action_values)
            best_action = torch.argmax(action_values, axis=1).item()
            #print("Selected action:", best_action)
            return best_action
# FrameSkipReplicator class modified from the original MaxAndSkipEnv wrapper to display all 4 frames (for nice video playback), but only return max_frame
from stable_baselines3.common.type_aliases import GymStepReturn
class FrameSkipReplicator(MaxAndSkipEnv):
    def __init__(self, env, skip=4, fps=60):
        super().__init__(env, skip)
        self.fps = fps
        self.skip = skip
    def step(self, action: int) -> GymStepReturn:
        total_reward = 0.0
        done = None
        last_frame_rendered = time.perf_counter()
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            current_time = time.perf_counter()
            while current_time - last_frame_rendered < 1/self.fps:
                current_time = time.perf_counter()
            last_frame_rendered = current_time
            env.render()
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info
##################################################
# Playback loop to see trained agent performance #
##################################################

# Seed the RNG for consistent sequence of random actions
random.seed(1)

# Setup a new environment without frameskipping (so that playback looks normal)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# Simplest moveset that can complete the level
NO_RUNNING = [
    ['right'],
    ['right', 'A']
]

# Specify what controller inputs are available
chosen_inputs = SIMPLE_MOVEMENT
env = JoypadSpace(env, chosen_inputs)

# Wrapper to ensure observations are in same format as was used in training
num_skip_frames = 4
game_fps = 60
env = FrameSkipReplicator(env, skip=num_skip_frames, fps=game_fps) # Replicate the frameskipping progress (one ouptut every n frames), but render all skipped frames
env = GrayScaleObservation(env) # Convert state from RGB to grayscale (1/3 number of pixels for NN to process)
env = ResizeObservation(env, shape=84) # Scale each frame to 64x64 pixels (15x fewer pixels for NN to process)
env = FrameStack(env, 4) # Stack the last 4 observations together to give NN temporal awareness

# Create a new agent using saved model
agent = Agent(len(chosen_inputs), model_path='./models/lr0.000025/20220419-122232/10000000.pth')
#agent.epsilon = 0.05 # Use this if you still want some degree of randomness in playback or to match training value
print(agent.epsilon)

steps_since_restart = 0
last_frame_rendered = time.perf_counter()

with torch.no_grad():
    #env.render()
    #time.sleep(5)
    for episode in range(100):
        state = env.reset()
        start_time = time.perf_counter()
        while True:
            with torch.no_grad():
                chosen_action = agent.get_action(state)
                # Perform n frames of the same action to account for the frameskipping used in training
                state, reward, done, info = env.step(chosen_action)

                #plt.imshow(state, cmap='gray')

                steps_since_restart += 1

                if done:
                    print("Episode", episode, end=': ')
                    if info['time'] == 0:
                        print("Mario ran out of time")
                    elif info['flag_get']:
                        print("########## Mario reached the FLAG in", time.perf_counter() - start_time, "seconds ##########")
                    else:
                        print("Mario died after", steps_since_restart, "steps at x_position", info['x_pos'])
                    break

env.close()
```

### The on-policy comparison algorithms: PPO

```python
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import os
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import configure
MARIO_ENV='SuperMarioBros-1-1-v0'

MODEL_DIR='models/'

####### new directories added
LOG_DIR = 'logs/'
new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


################################################################## For PPO trainning parameters ##################################################################
LEARNING_RATE = 2.5e-5
GAMMA = 0.9
LAMBDA = 0.9

MAX_STEPS = 15e6
#### Mario environments

def mario_env(train:bool = False) -> gym.Env:
    env = gym_super_mario_bros.make(MARIO_ENV)
    # Get action... can change into RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env, width=84, height=84)
    if train:
        return Monitor(env, LOG_DIR)
    else:
        return env
# ?PPO
model = PPO(
    "CnnPolicy",
    mario_env(train=True), 
    learning_rate=LEARNING_RATE,
    batch_size=32,
    gamma=GAMMA,
    gae_lambda=LAMBDA,
    create_eval_env=True,
    ent_coef=0.02,
    vf_coef=1.0,
    tensorboard_log=f'{LOG_DIR}_PPO_v2',
    verbose=1)

model.set_logger(new_logger)
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='checkpointcallback/')

eval_callback = EvalCallback(
    mario_env(train=True),
    best_model_save_path=f'./{MODEL_DIR}/best_model/',
    log_path=f'./{MODEL_DIR}/best_model/results/',
    eval_freq=5e4,
    deterministic=False,
    verbose=1)

callback = CallbackList([checkpoint_callback, eval_callback])


model.learn(total_timesteps=MAX_STEPS, callback=callback, reset_num_timesteps=False, tb_log_name='PPO_v2')
model.save(f'{MODEL_DIR}/{MAX_STEPS}')
```

### The on-policy comparison algorithms: A2C

```python
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import os
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import configure
MARIO_ENV='SuperMarioBros-1-1-v0'

MODEL_DIR='models/'

####### new directories added
LOG_DIR = 'logs/'
new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


################################################################## For PPO trainning parameters ##################################################################
LEARNING_RATE = 2.5e-5
GAMMA = 0.9
LAMBDA = 0.9

MAX_STEPS = 15e6
#### Mario environments

def mario_env(train:bool = False) -> gym.Env:
    env = gym_super_mario_bros.make(MARIO_ENV)
    # Get action... can change into RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env, width=84, height=84)
    if train:
        return Monitor(env, LOG_DIR)
    else:
        return env
# ?PPO
model = A2C(
    "CnnPolicy",
    mario_env(train=True), 
    learning_rate=LEARNING_RATE,
    n_steps=8,
    gamma=GAMMA,
    gae_lambda=LAMBDA,
    create_eval_env=True,
    ent_coef=0.02,
    vf_coef=1.0,
    tensorboard_log=LOG_DIR,
    verbose=1)

model.set_logger(new_logger)
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='checkpointcallback/')

eval_callback = EvalCallback(
    mario_env(train=True),
    best_model_save_path=f'./{MODEL_DIR}/best_model/',
    log_path=f'./{MODEL_DIR}/best_model/results/',
    eval_freq=5e4,
    deterministic=False,
    verbose=1)

callback = CallbackList([checkpoint_callback, eval_callback])


model.learn(total_timesteps=MAX_STEPS, callback=callback, reset_num_timesteps=False, tb_log_name='PPO')
model.save(f'{MODEL_DIR}/{MAX_STEPS}')
```