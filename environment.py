import gymnasium as gym
from utils import *
from model import Model
from agent import DQNAgent
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is being used: {device}")
env = gym.make("ALE/SpaceInvaders-v5")
observation, info = env.reset(seed=42)
LEARNING_RATE = 0.00015 #0.00025
NUM_OF_STACKED_FRAMES = 4
TRAINING_THRESHOLD = 500
EPSILON_DECAY = 0.00012
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.1
DISCOUNT_FACTOR = 0.99

action_value_function = Model(num_actions=env.action_space.n).to(device)
optimizer = torch.optim.RMSprop(action_value_function.parameters(), lr=LEARNING_RATE, momentum=0.95)
deep_q_agent = DQNAgent(EPSILON_DECAY, INITIAL_EPSILON, FINAL_EPSILON, DISCOUNT_FACTOR, action_value_function, torch.nn.MSELoss(), optimizer, device)
replay_buffer = ReplayBuffer(10000)
number_of_episodes = 10000

episode_scores = []

#Select Number of Episodes (TODO)
for ep in range(1, number_of_episodes+1):
    state = env.reset()
    terminated, truncated = False, False
    #total_reward = 0
    episode_score = 0
    current_state = torch.zeros((NUM_OF_STACKED_FRAMES, 84, 84), dtype=torch.float32, device=device)

    while not terminated and not truncated:
        previous_state = current_state

        """
        Run select_action on agent to decide on following:
        1. Use random action
        2. Use Deep-Q-Model Estimate
        """

        action = deep_q_agent.select_action(env.action_space.n, current_state.unsqueeze(0))

        observation, reward, terminated, truncated, info = env.step(action)
        episode_score += reward
        reward = max(-1, min(reward, 1))


        """
        Downsample frame to specified dimensions and convert frames to pytorch tensors
        """

        downsized_observation = downscale_image(observation)
        downsized_observation_tensor = torch.tensor(downsized_observation, dtype=torch.float32, device=device)

        current_state = update_frame_stack(current_state, downsized_observation_tensor)

        """
        Store data in instantiated Replay Memory
        """

        replay_buffer.insert_data(previous_state, action, reward, current_state, terminated, truncated)
        replay_buffer.update_replay_buffer()
 
        """
        Sample n amount of data from Replay Memory
        """

        batch_data = replay_buffer.sample_data(32)


        if replay_buffer.get_length() >= TRAINING_THRESHOLD:
            deep_q_agent.training_step(batch_data)
            del batch_data

    torch.cuda.empty_cache()

    deep_q_agent.decrement_epsilon()
    episode_scores.append(episode_score)
    print(f"Episode {ep} has ended. Agent has scored {episode_score} for this episode")
    print(f"Current Epsilon value is {deep_q_agent.get_epsilon()}")

# Assuming episode_scores is a list containing the score of each episode
average_scores = []
chunk_size = 100

for i in range(0, len(episode_scores), chunk_size):
    chunk = episode_scores[i:i + chunk_size]
    average_score = sum(chunk) / len(chunk)
    average_scores.append(average_score)

final_model = deep_q_agent.get_final_model()
# Save the entire model
torch.save(final_model, 'deep_q_agent_v2.pth')

# Plotting the average scores
plt.plot(average_scores)
plt.title("Average Scores per 100 Episodes")
plt.xlabel("Chunk of 100 Episodes")
plt.ylabel("Average Score")
plt.show()


env.close()