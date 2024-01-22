import environment

my_env = environment.TestEnvironment("LunarLander-v2", True)
my_env.start_episode()
my_env.start_episode_loop()