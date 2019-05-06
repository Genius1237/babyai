from babyai.rl.algos import PPOAlgo
from babyai.rl.utils import ParallelEnv
import babyai
import copy
import random
import time
import logging
import numpy as np

from multiprocessing import Process, Pipe
from multiprocessing.managers import BaseManager
import gym
import gc

import traceback
import sys
import math
import os
import pickle

class RCParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    @staticmethod
    def worker(conn, random_seed, curriculum_loc):
        good_start_states = pickle.load(open('{}_0.pickle'.format(curriculum_loc),'rb'))
        random.seed(random_seed)
        env = copy.deepcopy(random.choice(good_start_states))
        curr_no = 0
        while True:
            try:
                cmd, data = conn.recv()
                if cmd == "step":
                    obs, reward, done, info = env.step(data)
                    if done:
                        env = copy.deepcopy(random.choice(good_start_states))
                        obs = env.gen_obs()
                    conn.send((obs, reward, done, info))
                elif cmd == "reset":
                    env = copy.deepcopy(random.choice(good_start_states))
                    obs = env.gen_obs()
                    conn.send(obs)
                elif cmd == "print":
                    #print(env,env.mission)
                    conn.send(env.count)
                elif cmd == "update":
                    if data != curr_no:
                        good_start_states = pickle.load(open('{}_{}.pickle'.format(curriculum_loc,data),'rb'))
                        curr_no = data
                    conn.send("done")
                    continue

                else:
                    raise NotImplementedError
            except:
                traceback.print_exc()

    def __init__(self, env_name, n_env, curriculum_loc):
        assert n_env >= 1, "No environment given."

        self.env_name = env_name
        self.n_env = n_env
        temp_env = gym.make(env_name)
        self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space
        self.curriculum_loc = curriculum_loc

        self.locals = []
        self.processes = []
        rand_seed = random.randint(0,n_env-1)
        for i in range(self.n_env):
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=RCParallelEnv.worker, args=(remote, rand_seed+i, curriculum_loc))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        #self.envs[0].env = copy.deepcopy(random.choice(RCParallelEnv.good_start_states))
        #results = [self.envs[0].gen_obs()] + [local.recv() for local in self.locals]
        results = [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions):
            local.send(("step", action))
        #if done:
        #    self.envs[0].env = copy.deepcopy(random.choice(RCParallelEnv.good_start_states))
        #obs, reward, done, info = self.envs[0].step(actions[0])
        #    obs = self.envs[0].gen_obs()
        #results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        results = zip(*[local.recv() for local in self.locals])
        return results
    
    def print(self):
        for local in self.locals:
            local.send(("print",None))
        print(sum([local.recv() for local in self.locals])/len(self.locals))

    def __del__(self):
        for p in self.processes:
            p.terminate()

    def update_good_start_states(self,curr_no):
        #print(sys.getrefcount(good_start_states),sys.getsizeof(good_start_states))
        
        [local.send(("update",curr_no)) for i,local in enumerate(self.locals)]
        [local.recv() for local in self.locals]

def smooth(array, smoothing_weight):
    result = []
    prev = array[0]
    for x in array:
        curr = prev * smoothing_weight + (1-smoothing_weight)*x
        result.append(curr)
        prev = curr
    return result[-1]

class RCPPOAlgo(PPOAlgo):
    """
    The class containing an application of Reverse Curriculum learning from
    https://arxiv.org/pdf/1707.05300.pdf to Proximal Policy Optimization
    """
    def __init__(self, env_name, n_env, acmodel, demo_loc, update_frequency = 10, curr_method='1', curr_n_demos=500, tsc_method = 1,
                 tsc_alpha = 0.1, tsc_epsilon = 0.1, num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95, entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, aux_info=None):

        self.n_env = n_env
        self.env_name = env_name
        self.transfer_ratio = 0.15
        self.tsc_alpha = tsc_alpha
        self.tsc_epsilon = tsc_epsilon
        self.tsc_method = tsc_method
        self.curr_n_demos = curr_n_demos
        self.curr_method = curr_method
        self.update_frequency = update_frequency
        super().__init__([gym.make(env_name) for _ in range(n_env)], acmodel, num_frames_per_proc, discount, lr, beta1, beta2, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, adam_eps, clip_eps, epochs,
                         batch_size, preprocess_obss, reshape_reward, aux_info)
        self.env = None

        self.read_good_start_states(env_name,demo_loc,self.curr_n_demos,self.curr_method)
        
        self.env = RCParallelEnv(self.env_name,self.n_env,self.curriculum_loc)
        self.obs = self.env.reset()
        
        self.update = 0
        self.curr_update = 1
        
        self.curr_q = [0 for _ in range(self.curriculum_length)]
        if self.tsc_method == 1:
            self.curr_prev_reward = [0 for _ in range(self.curriculum_length)]
        elif self.tsc_method == 2:
            self.success_history = []

        self.log_history = []
        self.curr_done = False


    def update_parameters(self):
        logs = super().update_parameters()
        '''logs = {
            "entropy":0,"value":0,"policy_loss":0,"value_loss":0,"grad_norm":0,"loss":0,"return_per_episode": [0],"reshaped_return_per_episode": [0],"num_frames_per_episode": [0],"num_frames": 0,"episodes_done": 0
        }'''
        
        success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
        if self.tsc_method == 2:
            self.success_history.append(success_rate)
        self.update += 1
        
        if not self.curr_done and self.update % self.update_frequency == 0:
            #reward = np.mean([r for r in logs['return_per_episode']])
            self.log_history.append(success_rate)
            logger = logging.getLogger(__name__)

            if len(self.log_history)>=10 and smooth(self.log_history[-10:],0.8)>=0.9:
                self.curr_done = True
                logger.info('Curriculum Updates Done')
                self.env = ParallelEnv([gym.make(self.env_name) for _ in range(self.n_env)])
            else:
                if self.tsc_method == 1:
                    r = success_rate - self.curr_prev_reward[self.curr_chosen]
                    self.curr_prev_reward[self.curr_chosen] = success_rate
                
                elif self.tsc_method == 2:
                    r = np.polyfit([_ for _ in range(len(self.success_history))],self.success_history,1)[0]
                    self.success_history = []

                self.curr_q[self.curr_chosen] = self.tsc_alpha * r + (1-self.tsc_alpha)*self.curr_q[self.curr_chosen]
                
                # epsilon-greedy sampling
                p = random.random()
                if p > self.tsc_epsilon:
                    argmax = np.argmax(self.curr_q)
                    self.curr_chosen = argmax
                else:
                    self.curr_chosen = random.randint(0,self.curriculum_length-1)
                
                logger.info('Changing to curriculum {}'.format(self.curr_chosen))
                self.env.update_good_start_states(self.curr_chosen)
            
            #logger.info('Start state Update Number {}/{}'.format(self.curr_update,self.curriculum_length-1))
            
            '''
            if self.curr_update < self.curriculum_length:
                if True:
                #if self.early_stopping_check(min_delta,patience):
                    
                    self.env.update_good_start_states(self.curr_update)
                    self.curr_update+=1
                    logger.info('Start state Update Number {}/{}'.format(self.curr_update,self.curriculum_length-1))

            elif self.curr_update == self.curriculum_length:
                self.curr_update += 1
                logger.info('Start State Updates Done')
                self.env = ParallelEnv([gym.make(self.env_name) for _ in range(self.n_env)])
            '''
        return logs
   
    def read_good_start_states(self, env_name, demo_loc,curr_n_demos,curr_method):
        curr_method = int(curr_method)
        demos = babyai.utils.demos.load_demos(demo_loc)[:curr_n_demos]
        
        seed = 0
        max_len = max([len(demo[3]) for demo in demos]) -1
        n_stages = math.ceil((max_len+1)/curr_method)

        start_states = [[] for _ in range(n_stages)]

        for i,demo in enumerate(demos):
            actions = demo[3]

            env = gym.make(env_name)
            env.seed(seed+i)
            env.reset()
            env.count = len(actions)
            
            n_steps = len(actions) -1
            #0 0 .... 0 (max_len-n_steps) 0 to n_steps -1 (n_steps)

            for j in range(max_len-1,n_steps-1,-1):
                if random.randint(1,curr_method) == 1:
                    start_states[math.floor(j/curr_method)].append(copy.deepcopy(env))

            for j in range(n_steps-1,-1,-1):
                _,_,done,_ = env.step(actions[j].value)
                env.count -= 1
                env.step_count = 0
                if random.randint(1,curr_method) == 1:
                    start_states[math.floor(j/curr_method)].append(copy.deepcopy(env))

        save_dir = os.environ['SLURM_TMPDIR'] if 'SLURM_TMPDIR' in os.environ else '/tmp'

        self.curriculum_loc = os.path.join(save_dir,str(int(time.time())))

        for i in range(len(start_states)):
            save_loc = '{}_{}.pickle'.format(os.path.join(save_dir,self.curriculum_loc),i)
            pickle.dump(start_states[i],open(save_loc,'wb'))
        
        self.curriculum_length = n_stages
        self.curr_chosen = 0