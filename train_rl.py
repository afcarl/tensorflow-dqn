import time

import gym
import numpy as np
from tqdm import tqdm

from agent import setup_agent
from opt import parse_opt

def burn_in_memory(env, agent, opt):
    # Burn in for a number of steps
    state = env.reset()
    done = False
    
    for _ in range(opt['burn_in']):
        if opt['render']:
            env.render()
        # Interaction
        agent.update_epsilon(test=True) # Use minimum epsilon
        action = agent.step(state)
        next_state, reward, done, _ = env.step(action)

        agent.replaymemory.add(state, action, reward, next_state, done)    

        if done:
            state = env.reset()
            done = False
        else:
            state = next_state

def eval(env, agent, eval_episodes, render=False):
    
    rewards = []
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.
        while not done:     
            if render:
                env.render()
            agent.update_epsilon(test=True)
            action = agent.step(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
        rewards.append(episode_reward)
    return rewards

def main():
    
    opt = parse_opt()

    # Setup 
    env = gym.make(opt['env'])
    agent = setup_agent(env, opt)
    burn_in_memory(env, agent, opt)

    # Load if specified
    if opt['load_path'] is not None:
        agent.load(opt['load_path'])
    
    # Main Training Loop
    for step in tqdm(range(1, opt['max_steps']+1, 1)):
        if step == 1:
            state = env.reset()
            done = False

        if opt['render']:
            env.render()

        agent.update_epsilon(step)
        action = agent.step(state)
        next_state, reward, done, _ = env.step(action)
        
        # Add to replay memory and update network
        agent.replaymemory.add(state, action, reward, next_state, done)
        agent.update_network()

        if done:
            state = env.reset()
            done = False
        else:
            state = next_state

        if step % opt['freeze_interval'] == 0:
            agent.copy_qnetwork()

        if step % opt['eval_interval'] == 0:
            eval_rewards = eval(env, agent, opt['eval_episodes'])
            print("step", step, "average reward", "{}(+/-{})".format(np.mean(eval_rewards), np.std(eval_rewards)))
            # Reset to continue training
            env.reset()
        
        if step % opt['save_interval'] == 0:
            agent.save(opt['save_path'], global_step=step)

    # Save args


if __name__ == "__main__":
    main()