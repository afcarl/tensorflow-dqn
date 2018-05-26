import copy
import random

import numpy as np
from parlai.core.agents import Agent
import tensorflow as tf 

from .qnetwork import QNetwork
from .replaymemory import ReplayMemory
from .scheduler import LinearScheduler
from . import tf_utils as tf_utils

def setup_agent(env, opt):
    # Add env parameters to opt for qnetwork creation
    opt['input_size'] = env.observation_space.shape[0]
    opt['output_size'] = env.action_space.n
    agent = DQNAgent(opt)
    return agent

class DQNAgent(Agent):
    """
        The DQN Agent that interacts with the environment
        Does not manipulating tensorflow directly
        However, agent maintains Tensorflow session for running qnetwork
    """
    def __init__(self, opt, shared=None):
        super(DQNAgent, self).__init__(opt, shared)
        
        self.setup_qnetwork(opt)
        self.replaymemory = ReplayMemory(opt['memory_size'])
        self.scheduler = LinearScheduler(opt['init_epsilon'], opt['min_epsilon'], opt['schedule_timesteps'])

        self.batch_size = opt['batch_size']
        self.epsilon = opt['init_epsilon']

        # Initialize frozen network
        self.copy_qnetwork()

        # Load and save
        self.saver = tf_utils.create_saver()

    def setup_qnetwork(self, opt):
        # Create Session
        self.sess = tf_utils.create_session()
        # Create Q Network and Target Network
        self.qnetwork = QNetwork(opt, name='qnetwork')
        self.target_qnetwork = QNetwork(opt, name='target_qnetwork')  
        # Build Copy Operation
        self.copy_op = tf_utils.copy_variable_scope(source_name='qnetwork', target_name='target_qnetwork')
        # Initialize all variables
        tf_utils.initialize_all_variables(self.sess)    
       
    def copy_qnetwork(self):
        # Copy current Qnetwork to previous qnetwork
        #self.qnetwork.copy_to_target()
        pass

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
        
        q_values = self.qnetwork.predict_one_batch(self.sess, state) # (batch_size, action_space)
        
        # Here we decide which epsilon decay policy we should use
        action = self.epsilon_greedy_policy(q_values)
        
        return action

    def update_network(self):
        """
            Sample from Replay Memory and update network for one batch
        """
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_done = \
            self.replaymemory.encode_sample(self.batch_size)

        # Create Network Target Value
        #batch_target_qvalues = self.target_qnetwork.predict_one_batch(batch_next_states) #(batch_size, action_size)
        batch_target_qvalues = self.qnetwork.predict_one_batch(self.sess, batch_next_states)
        batch_max_target_qvalues = np.max(batch_target_qvalues,axis=-1) #(batch_size) , this is the target

        batch_target_qvalues = batch_rewards + (1 - batch_done) * batch_max_target_qvalues 

        # Pass to Qnetwork for update
        self.qnetwork.train_one_batch(self.sess, batch_states, batch_actions, batch_target_qvalues)
    
    def save(self, exp_path, global_step=None):
        self.saver.save(self.sess, exp_path, global_step)
    
    def load(self, load_path):
        print("Restoring from {}".format(load_path))
        self.saver.restore(self.sess, load_path)

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
