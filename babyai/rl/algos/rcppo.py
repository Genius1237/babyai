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
import os
import itertools
import math

def generator(env_name, demo_loc, curr_method):
    demos = babyai.utils.demos.load_demos(demo_loc)

    seed = 0
    max_len = max([len(demo[3]) for demo in demos]) -1
    envs = []
    for i in range(len(demos)):
        env = gym.make(env_name)
        env.seed(seed+i)
        env.reset()
        envs.append(env)

    states = []
    curr_done = False
    prob = 0
    for ll in range(max_len):
        if curr_method == 'log':
            prob += 2**ll
        else:   
            prob = int(curr_method)*len(demos)
        prob = min(prob,max_len)

        if ll == max_len - 1:
            curr_done=True

        for i,demo in enumerate(demos):
            actions = demo[3]
            
            env = copy.deepcopy(envs[i])
            
            n_steps = len(actions) -1
            
            for j in range(n_steps-ll):
                _,_,done,_ = env.step(actions[j].value)
            if random.randint(1,prob) == 1:
                states.append(env)
            env.step_count = 0
            env.count=0
        
        if curr_method == 'log':
            if math.log2(ll+2) == int(math.log2(ll+2)) or curr_done:
                yield states,curr_done
                states = []
        else:
            num = int(curr_method)
            if ll%num == num-1 or curr_done:
                yield states,curr_done
                states = []
        

def worker(conn, random_seed, env_name, demo_loc, curr_method):
    #babyai.utils.seed(0)
    random.seed(random_seed)
    
    start_state_generator = generator(env_name, demo_loc, curr_method)

    i=0
    #good_start_states = []
    for good_start_states,curr_done in start_state_generator:
        #good_start_states.extend(good_start_states_u)
        if i==0:
            i+=1
            env = copy.deepcopy(random.choice(good_start_states))
        else:
            if curr_done:
                conn.send("curr_done")
            else:
                conn.send("done")
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
                    break
                else:
                    raise NotImplementedError
            except:
                traceback.print_exc()


class RCParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""
    
    def __init__(self, env_name, n_env, demo_loc, curr_method):
        assert n_env >= 1, "No environment given."

        self.env_name = env_name
        self.n_env = n_env
        temp_env = gym.make(env_name)
        self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space

        self.locals = []
        self.processes = []
        rand_seed = random.randint(0,n_env-1)
        for i in range(self.n_env):
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, rand_seed+i, env_name, demo_loc,curr_method))
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

    def update_good_start_states(self):
        #print(sys.getrefcount(good_start_states),sys.getsizeof(good_start_states))
        [local.send(("update",None)) for local in self.locals]
        t = [local.recv() for local in self.locals]
        if t[0] == "curr_done":
            return True
        else:
            return False

class RCPPOAlgo(PPOAlgo):
    """
    The class containing an application of Reverse Curriculum learning from
    https://arxiv.org/pdf/1707.05300.pdf to Proximal Policy Optimization
    """
    def __init__(self, env_name, n_env, acmodel, demo_loc, version, es_method=2, update_frequency = 10, transfer_ratio=0.15, random_walk_length=1, curr_method = 'one',num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, aux_info=None):

        self.n_env = n_env
        self.env_name = env_name
        self.transfer_ratio = transfer_ratio
        self.random_walk_length = random_walk_length
        self.version = version
        self.update_frequency = update_frequency
        self.es_method = es_method
        super().__init__([gym.make(env_name) for _ in range(n_env)], acmodel, num_frames_per_proc, discount, lr, beta1, beta2, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, adam_eps, clip_eps, epochs,
                         batch_size, preprocess_obss, reshape_reward, aux_info)

        if version == "v1":
            self.good_start_states = self.read_good_start_states(env_name, demo_loc)
        elif version == "v2" or version == "v3":
            self.read_good_start_states_v2(env_name,demo_loc,curr_method)
        self.env = None
        self.env = RCParallelEnv(self.env_name,self.n_env, demo_loc, curr_method)
        self.obs = self.env.reset()
        
        self.update = 0
        self.curr_update = 1
        self.log_history = []
        self.es_max = -1
        self.es_pat = 0
        self.curr_done = False
        self.curr_really_done = False

    def early_stopping_check(self, method, bound):
        '''
        if len(self.log_history) < patience:
            return False
        else:
            for i in range(patience-1):
                if self.log_history[-1-i]-self.log_history[-2-i] >= min_delta:
                    return False
            return True
        '''
        '''
        if len(self.log_history) ==0 :
            return False
        else:
            for i in range(patience):
                if self.log_history[-1-i] >= 0.9:
                    continue
                else:
                    return False
            return True
        '''
        if self.log_history[-1] >= bound:
            return True
        else:
            return False
        '''
        if self.log_history[-1] - self.es_max > min_delta:
            self.es_max = self.log_history[-1]
            self.es_pat = 0
            self.best_weights = self.acmodel.state_dict()
            ans = False
            no = 0
        else:
            self.es_pat += 1
            if self.es_pat >= patience:
                self.es_max = -1
                self.es_pat = 0
                self.acmodel.load_state_dict(self.best_weights)
                ans = True
                no = 1
            else:
                ans = False
                no = 1
        #print(ans,no,self.es_pat,patience)
        return ans
        '''
    def update_parameters(self):            
        logs = super().update_parameters()
        '''logs = {
            "entropy":0,"value":0,"policy_loss":0,"value_loss":0,"grad_norm":0,"loss":0,"return_per_episode": [0],"reshaped_return_per_episode": [0],"num_frames_per_episode": [0],"num_frames": 0,"episodes_done": 0
        }'''
        
        self.update += 1
        
        if self.version == "v1":
            if self.update % self.update_frequency == 0 and self.update//self.update_frequency < 15:
                self.good_start_states = self.update_good_start_states(self.good_start_states,self.random_walk_length,self.transfer_ratio)
                self.env.update_good_start_states()
                for state in self.good_start_states[-3:]:
                    s1 = copy.copy(state)
                    s1.render()
                    input()

        elif self.version == "v2":
            logger = logging.getLogger(__name__)
            if self.update % self.update_frequency ==0 and self.update//self.update_frequency < self.curriculum_length:
                
                """self.env.print()
                print(sum([state.count for state in self.env.good_start_states])/len(self.env.good_start_states))"""
                self.env.update_good_start_states()
                logger.info('Start state Update Number {}/{}'.format(self.update//self.update_frequency,self.curriculum_length))

            if self.update % self.update_frequency ==0 and self.update//self.update_frequency == self.curriculum_length:
                logger.info('Start State Updates Done')
                self.env = ParallelEnv([gym.make(self.env_name) for _ in range(self.n_env)])
        
        elif self.version == "v3":
            if self.update % self.update_frequency == 0 and not self.curr_really_done:
                success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
                self.log_history.append(success_rate)
                logger = logging.getLogger(__name__)

                min_delta = 0.025
                patience = 1
                if self.es_method == 1:
                    bound = 0.9
                elif self.es_method == 2:
                    bound = 0.7+(self.curr_update/self.curriculum_length)*(0.99-0.7)

                if not self.curr_done:
                    #if self.early_stopping_check(patience+(self.curr_update),min_delta):
                    if self.early_stopping_check(self.es_method,bound):    
                        self.curr_update+=1
                        self.log_history = []
                        self.curr_done = self.env.update_good_start_states()
                        logger.info('Start state Update Number {}'.format(self.curr_update))
                
                else:
                    if self.early_stopping_check(self.es_method,bound):
                        self.curr_update += 1
                        self.log_history = []
                        logger.info('Start State Updates Done')
                        
                        self.env = ParallelEnv([gym.make(self.env_name) for _ in range(self.n_env)])
                        self.curr_really_done = True


        #self.obs = self.env.reset()

        return logs
        
    def update_good_start_states(self, good_start_states, random_walk_length, transfer_ratio):
        new_starts = []
        #new_starts.extend(copy.deepcopy(self.good_start_states))
        
        #"""
        for state in good_start_states:
            s1 = state
            for i in range(random_walk_length):
                s1 = copy.deepcopy(s1)
                action = s1.action_space.sample()
                s1.step(action)
                s1.count += 1
                s1.step_count = 0
            new_starts.append(s1)

        """
        #n_threads = self.n_env
        n_threads = 64
        for start in range(0,len(self.good_start_states),n_threads):
            end = min(start+n_threads,len(self.good_start_states))
            
            good_start_states = ParallelEnv(self.good_start_states[start:end])
            for i in range(n_explore):
                action = [good_start_states.action_space.sample() for _ in range(len(good_start_states.envs))]
                good_start_states.step(action)
                new_starts.extend(copy.deepcopy(good_start_states.envs))
        """

        n_old = int(transfer_ratio*len(good_start_states))
        l = len(good_start_states)
        good_start_states = random.sample(good_start_states,n_old)
        good_start_states.extend(random.sample(new_starts,l-n_old))
        
        return good_start_states

    def read_good_start_states(self,env_name,demo_loc):
        demos = babyai.utils.demos.load_demos(demo_loc)

        seed = 0
        start_states = []
        
        for i,demo in enumerate(demos):
            actions = demo[3]

            env = gym.make(env_name)
            
            babyai.utils.seed(seed)
            
            env.seed(seed+i)
            env.reset()
            for j in range(len(actions)-1):
                _,_,done,_ = env.step(actions[j].value)
            env.step_count = 0
            env.count = 1
            start_states.append(env)

        return start_states[:500]

    def read_good_start_states_v2(self, env_name, demo_loc,curr_method):
        demos = babyai.utils.demos.load_demos(demo_loc)

        seed = 0
        max_len = max([len(demo[3]) for demo in demos]) -1
        self.pos = 0
        if curr_method == 'log':
            self.curriculum_length = math.floor(math.log2(max_len)) + 1
        else:
            combining_factor = int(curr_method)
            self.curriculum_length = math.ceil(max_len/combining_factor)

        return
        
        self.start_states = [[] for _ in range(max_len)]

        for i,demo in enumerate(demos):
            actions = demo[3]

            env = gym.make(env_name)
            env.seed(seed+i)
            env.reset()
            env.count = len(actions)
            
            n_steps = len(actions) -1
            for j in range(max_len-1,n_steps-1,-1):
                self.start_states[j].append(copy.deepcopy(env))

            for j in range(n_steps):
                _,_,done,_ = env.step(actions[j].value)
                env.count -= 1
                env.step_count = 0
                self.start_states[n_steps-j-1].append(copy.deepcopy(env))


    def update_good_start_states_v2(self):
        self.pos += 1
        new_starts = self.start_states[self.pos]

        l = len(self.good_start_states)
        n_old = int(self.transfer_ratio*l)
        good_start_states = random.sample(self.good_start_states,n_old)
        good_start_states.extend(random.sample(new_starts,l-n_old))

        return good_start_states