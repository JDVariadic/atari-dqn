import gymnasium as gym

class TestEnvironment:
    def __init__(self, environment_name, is_show_game):
        self.env = gym.make(environment_name, render_mode="human") if is_show_game else gym.make(environment_name)
        self.current_observation = None
        self.current_info = None

    def start_episode(self):
        self.current_observation, self.current_info = self.env.reset()
    
    def start_episode_loop(self):
        for _ in range(1000):
            action = self.env.action_space.sample()
            self.current_observation, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            self.current_observation, self.current_info = self.env.reset()
