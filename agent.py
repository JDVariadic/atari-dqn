import copy
import random

class DQNAgent:
    def __init__(self, epsilon, epsilon_decay, initial_epsilon, final_epsilon, action_value_function):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.action_value_function = action_value_function
        self.target_action_value_function = copy.deepcopy(action_value_function)
        
    def select_action(self):
        if random.random() < self.epsilon:
                return "random action"
        return "non-random action"
    
    def update_action_value_function(self, new_action_value_function):
         self.action_value_function = new_action_value_function

    def update_target_action_Value_function(self):
         self.target_action_value_function = copy.deepcopy(self.action_value_function)