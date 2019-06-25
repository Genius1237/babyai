from babyai.rl.algos import PPOAlgo
from babyai.rl.utils import ParallelEnv
import babyai
import copy
import random
import logging
import numpy as np

from multiprocessing import Process, Pipe
import gym

import math

def generator(env_name, demo_loc, curr_method):
    demos = babyai.utils.demos.load_demos(demo_loc)

    seed = 0
    max_len = max([len(demo[3]) for demo in demos]) - 1
    envs = []
    for i in range(len(demos)):
        env = gym.make(env_name)
        env.seed(seed + i)
        env.reset()
        envs.append(env)

    states = []
    curr_done = False
    prob = 0
    prob_sum = 0
    int_curr_method = int(curr_method) if curr_method != 'log' else None
    for ll in range(max_len):
        if curr_method == 'log':
            log_var = math.log2(ll + 1)
            if log_var == int(log_var):
                prob = ll + 1
                prob_sum += prob
                if prob_sum >= max_len:
                    prob = max_len - prob_sum + (ll + 1)
                    prob_sum = max_len
        else:
            if ll % int_curr_method == 0:
                prob = int_curr_method
                prob_sum += int_curr_method
                if prob_sum >= max_len:
                    prob = max_len - prob_sum + int_curr_method
                    prob_sum = max_len

        if ll == max_len - 1:
            curr_done = True

        for i, demo in enumerate(demos):
            actions = demo[3]

            env = copy.deepcopy(envs[i])
            n_steps = len(actions) - 1

            for j in range(n_steps - ll):
                _, _, done, _ = env.step(actions[j].value)
            if random.randint(1, prob) == 1:
                states.append(env)
            env.step_count = 0
            env.count = 0

        if curr_method == 'log':
            if math.log2(ll + 2) == int(math.log2(ll + 2)) or curr_done:
                yield states, curr_done
                states = []
        else:
            if ll % int_curr_method == int_curr_method - 1 or curr_done:
                yield states, curr_done
                states = []


def worker(conn, random_seed, env_name, demo_loc, curr_method):
    # babyai.utils.seed(0)
    random.seed(random_seed)

    start_state_generator = generator(env_name, demo_loc, curr_method)

    i = 0
    # good_start_states = []
    for good_start_states, curr_done in start_state_generator:
        # good_start_states.extend(good_start_states_u)
        if i == 0:
            i += 1
            env = copy.deepcopy(random.choice(good_start_states))
        else:
            if curr_done:
                conn.send("curr_done")
            else:
                conn.send("done")
        while True:
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
                # print(env,env.mission)
                conn.send(env.count)
            elif cmd == "update":
                break
            else:
                raise NotImplementedError


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
        rand_seed = random.randint(0, n_env - 1)
        for i in range(self.n_env):
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, rand_seed + i, env_name, demo_loc, curr_method))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions):
            local.send(("step", action))
        results = zip(*[local.recv() for local in self.locals])
        return results

    def print(self):
        for local in self.locals:
            local.send(("print", None))
        print(sum([local.recv() for local in self.locals]) / len(self.locals))

    def __del__(self):
        for p in self.processes:
            p.terminate()

    def update_good_start_states(self):
        [local.send(("update", None)) for local in self.locals]
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
    def __init__(self, env_name, n_env, acmodel, demo_loc, es_method=2, update_frequency=10,
                 transfer_ratio=0.15, curr_method='one',
                 num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, aux_info=None):

        self.n_env = n_env
        self.env_name = env_name
        self.transfer_ratio = transfer_ratio
        self.update_frequency = update_frequency
        self.es_method = es_method
        super().__init__([gym.make(env_name) for _ in range(n_env)], acmodel, num_frames_per_proc, discount, lr, beta1,
                         beta2, gae_lambda, entropy_coef, value_loss_coef, max_grad_norm, recurrence, adam_eps,
                         clip_eps, epochs, batch_size, preprocess_obss, reshape_reward, aux_info)

        self.read_good_start_states(env_name, demo_loc, curr_method)
        self.env = None
        self.env = RCParallelEnv(self.env_name, self.n_env, demo_loc, curr_method)
        self.obs = self.env.reset()

        self.update = 0
        self.curr_update = 1
        self.log_history = []
        self.es_max = -1
        self.es_pat = 0
        self.curr_done = False
        self.curr_really_done = False

    def early_stopping_check(self, method, bound):
        if self.log_history[-1] >= bound:
            return True
        else:
            return False

    def update_parameters(self):
        logs = super().update_parameters()

        self.update += 1

        if self.update % self.update_frequency == 0 and not self.curr_really_done:
            success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
            self.log_history.append(success_rate)
            logger = logging.getLogger(__name__)

            min_delta = 0.025
            patience = 1
            if self.es_method == 1:
                bound = 0.9
            elif self.es_method == 2:
                bound = 0.7 + (self.curr_update / self.curriculum_length) * (0.99 - 0.7)

            if not self.curr_done:
                # if self.early_stopping_check(patience+(self.curr_update),min_delta):
                if self.early_stopping_check(self.es_method, bound):
                    self.curr_update += 1
                    self.log_history = []
                    self.curr_done = self.env.update_good_start_states()
                    logger.info('Start state Update Number {}'.format(self.curr_update))

            else:
                if self.early_stopping_check(self.es_method, bound):
                    self.curr_update += 1
                    self.log_history = []
                    logger.info('Start State Updates Done')

                    self.env = ParallelEnv([gym.make(self.env_name) for _ in range(self.n_env)])
                    self.curr_really_done = True

        # self.obs = self.env.reset()

        return logs

    def read_good_start_states(self, env_name, demo_loc, curr_method):
        demos = babyai.utils.demos.load_demos(demo_loc)

        seed = 0
        max_len = max([len(demo[3]) for demo in demos]) - 1
        self.pos = 0
        if curr_method == 'log':
            self.curriculum_length = math.floor(math.log2(max_len)) + 1
        else:
            combining_factor = int(curr_method)
            self.curriculum_length = math.ceil(max_len / combining_factor)
