import time

import gym

from agent import DQNAgent
from qnetwork import QNetwork
from opt import parse_opt
from scheduler import LinearScheduler
from replaymemory import ReplayMemory

def setup_env(opt):
    env = gym.make(opt['env'])
    return env

def setup_agent(opt, env):
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    
    qnetwork = QNetwork(input_size, output_size)
    replaymemory = ReplayMemory(opt['memory_size'])
    
    agent = DQNAgent(qnetwork, replaymemory, opt)

    return agent

def run_episode(env, agent, render=False):
    
    state = env.reset()
    done = False
    while not done:
        
        if render:
            env.render()

        action = agent.step(state)
        next_state, reward, done, _ = env.step(action)
        
        # Add to replay memory
        agent.replaymemory.add(state, action, reward, next_state, done)

def main():
    
    opt = parse_opt()

    env = setup_env(opt)
    agent = setup_agent(opt, env)

    num_episodes = opt['num_episodes']
    import pdb; pdb.set_trace()
    for episode in range(1, num_episodes+1):
        print("Episode",episode)
        run_episode(env, agent, render=opt['render'])

if __name__ == "__main__":
    main()