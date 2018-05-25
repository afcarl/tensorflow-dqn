import copy
import random

import numpy as np
from parlai.core.agents import Agent

from .qnetwork import QNetwork
from .replaymemory import ReplayMemory
from .scheduler import LinearScheduler

def setup_agent(env, opt):
    opt['input_size'] = env.observation_space.shape[0]
    opt['output_size'] = env.action_space.n

    agent = DQNAgent(opt)

    return agent

class DQNAgent(Agent):
    def __init__(self, opt, shared=None):
        super(DQNAgent, self).__init__(opt, shared)
        
        self.qnetwork = QNetwork(opt['input_size'], opt['output_size'])
        self.replaymemory = ReplayMemory(opt['memory_size'])
        self.scheduler = LinearScheduler(opt['init_epsilon'], opt['min_epsilon'], opt['schedule_timesteps'])

        self.batch_size = opt['batch_size']
        self.epsilon = opt['init_epsilon']

        # Initialize frozen network
        self.copy_network()

    def copy_network(self):
        # Copy current Qnetwork to previous qnetwork
        self.next_qnetwork = self.qnetwork
        #self.next_qnetwork = copy.deepcopy(self.qnetwork)

    """ Reinforcement Learning related """
    def update_epsilon(self, current_timestep=None, test=False):
        if not test:
            self.epsilon = self.scheduler.value(current_timestep)
        else:
            self.epsilon = self.scheduler.end_value()
        return self.epsilon
    
    def greedy_policy(self, qvalues):
        """
            Select the action with maximum qvalue
        """
        action = np.argmax(qvalues)
        return action

    def epsilon_greedy_policy(self, qvalues, epsilon=None):
        """
            Sample according to probability
        """
        action_size = qvalues.size
        
        max_action_index = np.argmax(qvalues)
        probs = []
        for action_index in range(action_size):
            if action_index == max_action_index:
                action_prob = 1 - self.epsilon
            else:
                action_prob = self.epsilon / ( action_size - 1 )
            probs.append(action_prob)

        action = np.random.choice(action_size, p=probs)
        return action

    def step(self, state):
        """
            args:
            - state: numpy array

            return: 
            - action: int
        """
        state = np.expand_dims(state,axis=0) # Expect only one dimension only
        
        q_values = self.qnetwork.predict_one_batch(state) # (batch_size, action_space)
        
        # Here we decide which epsilon decay policy we should use
        action = self.epsilon_greedy_policy(q_values)
        
        return action

    def update_network(self):
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_done = \
            self.replaymemory.encode_sample(self.batch_size)

        # Create Network Target Value
        batch_next_qvalues = self.next_qnetwork.predict_one_batch(batch_next_states) #(batch_size, action_size)
        batch_max_next_qvalues = np.max(batch_next_qvalues,axis=-1) #(batch_size) , this is the target

        batch_target_qvalues = batch_rewards + (1 - batch_done) * batch_max_next_qvalues 

        # Pass to Qnetwork for update
        self.qnetwork.train_one_batch(batch_states, batch_actions, batch_target_qvalues)
    
    """ TODO: ParlAI related """
    def observe(self, observation):
        observation = copy.deepcopy(observation)
        self.observation = observation
        return self.observation

    def act(self):
        # Act according to observation
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
    

if __name__ == "__main__":
    from qnetwork import QNetwork