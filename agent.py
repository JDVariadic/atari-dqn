import torch
import copy
import random

class DQNAgent:
    def __init__(self, epsilon, epsilon_decay, initial_epsilon, final_epsilon, discount_factor, action_value_function, loss_fn, optimizer):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.action_value_function = action_value_function
        self.target_action_value_function = copy.deepcopy(action_value_function)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.training_step_counter = 0
        self.action_counter = 0
        self.previous_action = None
        
    def select_action(self, action_space, current_state):
        possible_actions = list(range(action_space))
        if self.action_counter % 4 == 0 or self.previous_action == None:
            if random.random() < self.epsilon:
                    chosen_action = random.choice(possible_actions)
            else:
                with torch.no_grad():
                    q_values = self.action_value_function(current_state)
                    chosen_action = torch.argmax(q_values).item()
            self.previous_action = chosen_action
        else:
            chosen_action = self.previous_action
        
        self.action_counter += 1
        return chosen_action
    
    def update_action_value_function(self, new_action_value_function):
        self.action_value_function = new_action_value_function

    def update_target_action_Value_function(self):
        self.target_action_value_function = copy.deepcopy(self.action_value_function)

    def training_step(self, minibatch):
        batch_states = minibatch[0][0] #[el[0][0] for el in minibatch]
        batch_actions = minibatch[0][1]#[el[1] for el in minibatch]
        batch_rewards = minibatch[0][2]#[el[2] for el in minibatch]
        batch_next_states= minibatch[0][3]#[el[3] for el in minibatch]
        batch_terminated = minibatch[0][4]#[el[4] for el in minibatch]
        batch_truncated = minibatch[0][5]#[el[5] for el in minibatch]
        print(len(batch_states))
        current_q = self.action_value_function(batch_states)[batch_actions]
        batch_total = current_q

        not_done_mask = ~ (batch_terminated | batch_truncated)
        next_q_max = torch.max(self.target_action_value_function(batch_next_states))
        batch_target = batch_rewards + self.discount_factor * next_q_max * not_done_mask
        loss = self.loss_fn(batch_total, batch_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.training_step_counter >= 1000:
            self.target_action_value_function = copy.deepcopy(self.action_value_function)
            self.training_step_counter = 0
        self.training_step_counter += 1
