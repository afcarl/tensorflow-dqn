import copy
import random

import numpy as np
from parlai.core.agents import Agent

class DQNAgent(Agent):
    def __init__(self, qnetwork, replaymemory, opt, shared=None):
        super(DQNAgent, self).__init__(opt, shared)
        self.qnetwork = qnetwork
        self.copy_network()

        self.replaymemory = replaymemory

        self.batch_size = opt['batch_size']

    def copy_network(self):
        # Copy current Qnetwork to previous qnetwork
        self.next_qnetwork = copy.deepcopy(self.qnetwork)

    """ Reinforcement Learning related """
    def step(self, state):
        """
            args:
            - state: numpy array

            return: 
            - action: int
        """
  
        state = np.expand_dims(state,axis=0)
        
        q_values = self.qnetwork.predict_one_batch(state) # (batch_size, action_space)
        
        # Here we decide which epsilon decay policy we should use
        action = random.choice(range(2))
        #action = self.greedy_policy(q_values)
        
        return action

    def greedy_policy(self, q_values):
        action = np.argmax(q_values)
        return action

    def update_network(self):
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_done = \
            self.replaymemory.encode_sample(self.batch_size)

        # Create Network Target Value
        batch_next_qvalues = self.next_qnetwork.predict_one_batch(batch_next_states) #(batch_size, action_size)
        batch_max_next_qvalues = np.max(batch_next_qvalues,axis=-1) #(batch_size) , this is the target

        batch_target_qvalues = batch_rewards + (1 - batch_done).reshape(self.batch_size,-1) * batch_max_next_qvalues 

        self.qnetwork.train_one_batch(batch_states, batch_actions, batch_target_qvalues)
    
    """ TODO: ParlAI related """
    def observe(self, observation):
        observation = copy.deepcopy(observation)
        self.observation = observation
        return self.observation

    def act(self):
        # Act according to observation
        pass

    def reset(self):
        pass
    