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

#Todo: Try moving replay to CPU and just do processing (unclear as we need to select action first)
#VECTORIZE TRAINING LOOP

EPSILON = 0.005
CAPACITY = 1000000
NUM_OF_SAMPLES = 32
DISCOUNT_FACTOR = 0.99
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.1
EPSILON_DECAY = 0.00003
LEARNING_RATE = 0.000025
NUM_OF_STEPS_TO_UPDATE = 10000
NUMBER_OF_FRAMES_PER_ACTION = 4
NUMBER_OF_GATHERED_DATA = 100

previous_action = None
steps = 0
collected_data = 0

current_epsilon = INITIAL_EPSILON
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
env = gym.make("ALE/SpaceInvaders-v5")

observation, info = env.reset()

max_episodes = 10000

replay_memory = deque()
action_value_function = Model().to(device)
target_action_value_function = copy.deepcopy(action_value_function).to(device)
optimizer = torch.optim.RMSprop(action_value_function.parameters(), lr=LEARNING_RATE, momentum=0.95)
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
    frame_stack = torch.tensor(frame_stack, dtype=torch.float32, device=torch.device(device))
    episode_score = 0
    previous_action = env.action_space.sample()
    current_frame_cycle = 1

    while not terminated and not truncated:
        previous_frame = copy.deepcopy(frame_stack).to(device)
        is_collecting_data = (collected_data < NUMBER_OF_GATHERED_DATA)
        if current_frame_cycle < NUMBER_OF_FRAMES_PER_ACTION:
            #print("Previous Action chosen")
            action = previous_action
            current_frame_cycle += 1
        else:
            if random.random() < EPSILON or (is_collecting_data):
                action = env.action_space.sample()
            else:
                action = torch.argmax(action_value_function(frame_stack)).item()
                #action = torch.argmax(action_value_function(frame_stack))[0]

            current_frame_cycle = 1
            previous_action = action
        
        

        observation, reward, terminated, truncated, info = env.step(action)
        episode_score += reward
        reward = max(-1, min(1, reward))
        

        processed_frame = torch.tensor(downscale_image(observation), dtype=torch.float32, device=torch.device(device))
        frame_stack = update_frame_stack(frame_stack, processed_frame)

        replay_memory.append((previous_frame, action, reward, frame_stack, terminated, truncated))
        collected_data += 1
        

        minibatch = None
        batch_target = None
        batch_total = None

        if len(replay_memory) < NUM_OF_SAMPLES:
            batch_target = torch.empty(len(replay_memory), dtype=torch.float16, device=torch.device(device))
            batch_total = torch.empty(len(replay_memory), dtype=torch.float16, device=torch.device(device))
            minibatch = random.sample(replay_memory, len(replay_memory))
        else:
            batch_target = torch.empty(NUM_OF_SAMPLES, dtype=torch.float16, device=torch.device(device))
            batch_total = torch.empty(NUM_OF_SAMPLES, dtype=torch.float16, device=torch.device(device))
            minibatch = random.sample(replay_memory, NUM_OF_SAMPLES)

        #Start of Computations
        batch_previous_frame = minibatch[0][0]
        batch_action = minibatch[0][1]
        batch_reward = minibatch[0][2]
        batch_frame_stack = minibatch[0][3]
        batch_terminated = minibatch[0][4]
        batch_truncated = minibatch[0][5]
        
        # Current Q-value for the action taken
        current_q = action_value_function(batch_previous_frame)[batch_action]
        batch_total = current_q

        not_done_mask = ~ (batch_terminated | batch_truncated)
        # Target Q-value calculation
        
        next_q_max = torch.max(target_action_value_function(batch_frame_stack))
        batch_target = batch_reward + DISCOUNT_FACTOR * next_q_max * not_done_mask
        #End of computations
        loss = loss_fn(batch_total, batch_target)

        #optimizer.zero_grad()
        for param in action_value_function.parameters():
            param.grad = None

        loss.backward()
        optimizer.step()
        
        if steps >= NUM_OF_STEPS_TO_UPDATE:
            target_action_value_function = copy.deepcopy(action_value_function).to(device)
            steps = 0
        else:
            steps += 1

        if len(replay_memory) > CAPACITY:
            replay_memory.popleft()

        if terminated or truncated:
            print(f"Episode {ep} ended. Resetting environment.")

    if (current_epsilon > FINAL_EPSILON) and (not is_collecting_data):
        current_epsilon -= EPSILON_DECAY
    print(f"Episode {ep} is finished with score {episode_score}")
    print(f"Current epsilon value is {current_epsilon}")
    episode_scores.append(episode_score)


env.close()

# Assuming episode_scores is a list containing the score of each episode
average_scores = []
chunk_size = 100

for i in range(0, len(episode_scores), chunk_size):
    chunk = episode_scores[i:i + chunk_size]
    average_score = sum(chunk) / len(chunk)
    average_scores.append(average_score)

# Plotting the average scores
plt.plot(average_scores)
plt.title("Average Scores per 100 Episodes")
plt.xlabel("Chunk of 100 Episodes")
plt.ylabel("Average Score")
plt.show()

