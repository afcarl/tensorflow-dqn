
class LinearScheduler(object):
    def __init__(self, init_epsilon, min_epsilon, schedule_timesteps):
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.schedule_timesteps = schedule_timesteps
    
    def value(self, t):
        """ Value of the schedule at time t"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.init_epsilon + fraction * (self.min_epsilon - self.init_epsilon)

    def end_value(self):
        return self.value(self.schedule_timesteps)
        
if __name__ == "__main__":
    sch = LinearScheduler(1.0, 0.0, 10)

    for t in range(10):
        print('value',sch.value(t))