import argparse

def parse_opt():
    parser = argparse.ArgumentParser("Debug DQN training")
    parser.add_argument('-env',dest='env',type=str,default='CartPole-v0',help='Gym environment name')
    parser.add_argument('-num-episodes',dest='num_episodes',type=int,default=10,help='Number of episodes to train')
    parser.add_argument('-num-steps',dest='num_steps',type=int,default=1e4,help='Number of steps to train')
    parser.add_argument('-memory-size',dest='memory_size',type=int,default=50000,help='Size of replay memory')
    parser.add_argument('-burn-in',dest='burn_in',type=int,default=1000,help='Number of experience steps to collect before learning starts')
    parser.add_argument('-batch-size',dest='batch_size',type=int,default=32,help='Batch size for training qnetwork')
    parser.add_argument('-render',action="store_true",default=False,help='Render gym environment')
    parser.add_argument('-init-epsilon',dest='init_epsilon',type=float,default=1.0,help='Initial exploration parameter')
    parser.add_argument('-min-epsilon',dest='min_epsilon',type=float,default=0.1,help='Final exploration parameter')
    parser.add_argument('-freeze-interval',dest='freeze_interval',type=int,default=500,help='Number of steps to freeze qnetwork')
    args = parser.parse_args()
    opt = vars(args)
    return opt

if __name__ == "__main__":
    opt = parse_opt()