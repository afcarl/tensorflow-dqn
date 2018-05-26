import argparse
import os

def parse_opt():
    parser = argparse.ArgumentParser("DQN training")
    parser.add_argument('-env',dest='env',type=str,default='CartPole-v0',help='Gym environment name')
    parser.add_argument('-max-steps',dest='max_steps',type=int,default=30000,help='Maximum of steps to train')
    parser.add_argument('-memory-size',dest='memory_size',type=int,default=50000,help='Size of replay memory')
    parser.add_argument('-burn-in',dest='burn_in',type=int,default=1000,help='Number of experience steps to collect before learning starts')
    parser.add_argument('-hidden-size',dest='hidden_size',type=int,default=100,help='Hidden size of QNetwork')
    parser.add_argument('-learning-rate',dest='learning_rate',type=float,default=5e-4,help='Learning rate for QNetwork optimizer')
    parser.add_argument('-batch-size',dest='batch_size',type=int,default=32,help='Batch size for training qnetwork')
    parser.add_argument('-render',action="store_true",default=False,help='Render gym environment')
    parser.add_argument('-init-epsilon',dest='init_epsilon',type=float,default=0.5,help='Initial exploration parameter')
    parser.add_argument('-min-epsilon',dest='min_epsilon',type=float,default=0.05,help='Final exploration parameter')
    parser.add_argument('-schedule-timesteps',dest='schedule_timesteps',type=int,default=50000,help='Timesteps from init_epsilon to min_epsilon')
    parser.add_argument('-freeze-interval',dest='freeze_interval',type=int,default=500,help='Number of steps to freeze qnetwork')
    parser.add_argument('-eval-interval',dest='eval_interval',type=int,default=500,help='Number of steps for evaluation')
    parser.add_argument('-eval-episode',dest='eval_episodes',type=int,default=100,help='Number of episodes for evaluation')
    parser.add_argument('-save-path',dest='save_path',type=str,default='./exp/tmp.ckpt',help='Path to save experiment')
    parser.add_argument('-load-path',dest='load_path',type=str,default=None,help='Path to load experiment')
    parser.add_argument('-save-interval',dest='save_interval',type=int,default=1000,help='Number of steps for every model save')
    args = parser.parse_args()
    opt = vars(args)
    return opt

if __name__ == "__main__":
    opt = parse_opt()