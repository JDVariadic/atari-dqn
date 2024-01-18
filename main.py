import gymnasium as gym
import pickle

from PIL import Image
import numpy as np

from utils import update_frame_stack
from utils import downscale_image

from collections import deque

from model import Model
import torch

import random
import copy

import matplotlib.pyplot as plt

EPSILON = 0.005
CAPACITY = 10000
NUM_OF_SAMPLES = 32
DISCOUNT_FACTOR = 99
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.1
EPSILON_DECAY = 0.003
LEARNING_RATE = 0.00025
NUM_OF_STEPS_TO_UPDATE = 10000
NUMBER_OF_FRAMES_PER_ACTION = 4

previous_action = None
steps = 0

current_epsilon = INITIAL_EPSILON

env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
#env = gym.make("ALE/SpaceInvaders-v5")

observation, info = env.reset()

max_episodes = 1000

replay_memory = deque()
action_value_function = Model()
target_action_value_function = copy.deepcopy(action_value_function)
optimizer = torch.optim.RMSprop(action_value_function.parameters(), lr=LEARNING_RATE, momentum=0.9)
loss_fn = torch.nn.MSELoss()

sampled_data = []
episode_scores = []
for ep in range(1, max_episodes+1):
    print(f"Starting episode {ep}")
    observation, info = env.reset()
    terminated = False
    truncated = False
    previous_frame = None
    reward = None
    frame_stack = np.zeros((4, 84, 84))
    frame_stack = torch.tensor(frame_stack, dtype=torch.float32)
    if torch.cuda.is_available():
        frame_stack = frame_stack
    episode_score = 0
    previous_action = env.action_space.sample()
    current_frame_cycle = 1

    while not terminated and not truncated:
        previous_frame = copy.deepcopy(frame_stack)
        if current_frame_cycle < NUMBER_OF_FRAMES_PER_ACTION:
            action = previous_action
            current_frame_cycle += 1
        else:
            if random.uniform(0, 1) >= EPSILON:
                action = env.action_space.sample()
            else:
                action = torch.argmax(action_value_function(frame_stack)).item()

            current_frame_cycle = 1
            previous_action = action
        
        

        observation, reward, terminated, truncated, info = env.step(action)
        episode_score += reward
        reward = max(-1, min(1, reward))
        

        processed_frame = torch.tensor(downscale_image(observation), dtype=torch.float32)
        frame_stack = update_frame_stack(frame_stack, processed_frame)

        replay_memory.append((previous_frame, action, reward, frame_stack, terminated, truncated))

        minibatch = None
        batch_target = None
        batch_total = None

        if len(replay_memory) < NUM_OF_SAMPLES:
            batch_target = torch.empty(len(replay_memory), dtype=torch.float)
            batch_total = torch.empty(len(replay_memory), dtype=torch.float)
            minibatch = random.sample(replay_memory, len(replay_memory))
        else:
            batch_target = torch.empty(NUM_OF_SAMPLES, dtype=torch.float)
            batch_total = torch.empty(NUM_OF_SAMPLES, dtype=torch.float)
            minibatch = random.sample(replay_memory, NUM_OF_SAMPLES)

        
        #SAMPLE DATA and compute targets
        for idx, transition in enumerate(minibatch):
            sampled_previous_frame, sampled_action, sampled_reward, sampled_frame_stack, sampled_terminated, sampled_truncated = transition
            
            prev_frame_tensor = sampled_previous_frame
            next_frame_tensor = sampled_frame_stack

            # Current Q-value for the action taken
            current_q = action_value_function(prev_frame_tensor)[sampled_action]
            batch_total[idx] = current_q

            # Target Q-value calculation
            if sampled_terminated or sampled_truncated:
                batch_target[idx] = sampled_reward
            else:
                next_q_max = torch.max(target_action_value_function(next_frame_tensor))
                batch_target[idx] = sampled_reward + DISCOUNT_FACTOR * next_q_max

        loss = loss_fn(batch_total, batch_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if steps >= NUM_OF_STEPS_TO_UPDATE:
            target_action_value_function = copy.deepcopy(action_value_function)

        if len(replay_memory) > CAPACITY:
            replay_memory.popleft()

        if terminated or truncated:
            print(f"Episode {ep} ended. Resetting environment.")

    if current_epsilon > FINAL_EPSILON:
        current_epsilon -= EPSILON_DECAY
    print(f"Episode {ep} is finished with score {episode_score}")
    print(f"Current epsilon value is {current_epsilon}")
    episode_scores.append(episode_score)


env.close()

plt.plot(episode_scores)
plt.title("Scores per Episode")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()


