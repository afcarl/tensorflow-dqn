import time

import gym

if __name__ == "__main__":
    
    env = gym.make("Pong-v0")

    state = env.reset()
    done = False
    
    while not done:
        
        action = env.action_space.sample()
        ob, reward, done, info = env.step(action)
        env.render()

        time.sleep(0.5)

