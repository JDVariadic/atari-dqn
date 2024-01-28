import gymnasium as gym
from utils import *
from model import Model
from agent import DQNAgent
from replay_buffer import ReplayBuffer

env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
observation, info = env.reset(seed=42)
LEARNING_RATE = 0.000025
NUM_OF_STACKED_FRAMES = 4

action_value_function = Model(num_actions=env.action_space.n)
optimizer = torch.optim.RMSprop(action_value_function.parameters(), lr=LEARNING_RATE, momentum=0.95)
deep_q_agent = DQNAgent(0.005, 0.00003, 1, 0.1, 0.99, action_value_function, torch.nn.MSELoss(), optimizer)
replay_buffer = ReplayBuffer(10000)
number_of_episodes = 1000

#Select Number of Episodes (TODO)
for ep in range(1, number_of_episodes+1):
    state = env.reset()
    terminated, truncated = False, False
    total_reward = 0
    current_state = torch.tensor(np.zeros((NUM_OF_STACKED_FRAMES, 84, 84)), dtype=torch.float32)

    while not terminated and not truncated:
        previous_state = current_state
        #Select Action (TODO)
        """
        Run select_action on agent to decide on following:
        1. Use random action
        2. Use Deep-Q-Model Estimate
        """
        #action = env.action_space.sample()  # this is where you would insert your policy
        action = deep_q_agent.select_action(env.action_space.n, current_state)

        #Observe Environment (TODO)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        reward = max(-1, min(reward, 1))

        #Process Data (TODO)
        """
        Downsample frame to specified dimensions and convert frames to pytorch tensors
        """

        downsized_observation = downscale_image(observation)
        downsized_observation_tensor = torch.tensor(downsized_observation, dtype=torch.float32)

        current_state = update_frame_stack(current_state, downsized_observation_tensor)
        #Store Data (TODO)
        """
        Store data in instantiated Replay Memory
        """

        replay_buffer.insert_data(previous_state, action, reward, current_state, terminated, truncated)

        #SAMPLE DATA (TODO)
        """
        Sample n amount of data from Replay Memory
        """

        batch_data = replay_buffer.sample_data(32)

        #TRAIN DATA (TODO)
        deep_q_agent.training_step(batch_data)

    print(f"Episode {ep} has ended. Agent has scored {total_reward} for this episode")

env.close()