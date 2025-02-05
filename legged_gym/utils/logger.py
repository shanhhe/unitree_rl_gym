import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
from torch.utils.tensorboard import SummaryWriter
import datetime
import os


class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None
        self.log_dir = '/home/shanhe/unitree_rl_gym/legged_gym/data/bruce'
        self.writer = None

    def log_state(self, key, value):
        print(key)
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()   

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()

    def plot_states(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(self.log_dir, timestamp)
        if self.log_dir is not None and self.writer is None:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        for key, values in self.state_log.items():
            for it, value in enumerate(values):
                if key == 'contact_forces_z':
                    self.writer.add_scalar(f'{key}/left', value[0], it)
                    self.writer.add_scalar(f'{key}/right', value[1], it)
                elif 'base' in key:
                    self.writer.add_scalar(f'Base/{key}', value, it)
                elif 'target' in key:
                    self.writer.add_scalar(f'Target/{key}', value, it)
                elif 'torque' in key:
                    self.writer.add_scalar(f'Torque/{key}', value, it)
                elif 'pos' in key:
                    self.writer.add_scalar(f'Pos/{key}', value, it)
                elif 'vel' in key:
                    self.writer.add_scalar(f'Vel/{key}', value, it)
                elif 'command' in key:
                    self.writer.add_scalar(f'Command/{key}', value, it)
                else:
                    self.writer.add_scalar(f'State/{key}', value, it)
        self.writer.close()
        

