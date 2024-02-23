import multiprocessing as mp
import os
import time

import numpy as np
import torch
from progressbar import ProgressBar
from torch.autograd import Variable
from torch.utils.data import Dataset
from socket import gethostname

from physics_engine_y import RopeEngine
from physics_engine_y import sample_init_p_flight

from utils import rand_float, rand_int, calc_dis
from utils import init_stat, combine_stat, load_data, store_data

np.random.seed(42)
# ======================================================================================================================
def normalize(data, stat, var=False):
    for i in range(len(stat)):
        stat[i][stat[i][:, 1] == 0, 1] = 1.0
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).to(data[i].device))
            data[i] = (data[i] - s[:, 0]) / s[:, 1]
    else:
        for i in range(len(stat)):
            # if data[i] is None or stat[i][:, 0] is None:
            #     print(f"None value found at index {i}\n")
            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]
    return data


def denormalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i])).to(data[i].device)
            data[i] = data[i] * s[:, 1] + s[:, 0]
    else:
        for i in range(len(stat)):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]
    return data


# ======================================================================================================================

def prepare_input(data, stat, args, param=None, var=False):
    if args.env == 'Rope':
        data = normalize(data, stat, var)
        attrs, states, actions = data

        # print('attrs', attrs.shape, np.mean(attrs), np.std(attrs))
        # print('states', states.shape, np.mean(states), np.std(states))
        # print('acts', acts.shape, np.mean(actions), np.std(actions))

        N = len(attrs)

        # print('N', N)

        rel_attrs = np.zeros((N, N, args.relation_dim))

        '''relation #0 self: root <- root'''
        rel_attrs[0, 0, 0] = 1

        '''relation #1 spring: root <- child'''
        rel_attrs[0, 1, 1] = 1

        '''relation #2 spring: child <- root'''
        rel_attrs[1, 0, 2] = 1

        '''relation #3 spring bihop: root <- child'''
        rel_attrs[0, 2, 3] = 1

        '''relation #4 spring bihop: child <- root'''
        rel_attrs[2, 0, 4] = 1

        '''relation #5 spring: child <- child'''
        for i in range(1, N - 1):
            rel_attrs[i, i + 1, 5] = rel_attrs[i + 1, i, 5] = 1

        '''relation #6 spring bihop: child <- child'''
        for i in range(1, N - 2):
            rel_attrs[i, i + 2, 6] = rel_attrs[i + 2, i, 6] = 1

        '''relation #7 self: child <- child'''
        np.fill_diagonal(rel_attrs[1:, 1:, 7], 1)

        assert (rel_attrs.sum(2) <= 1).all()

        # check the number of each edge type
        rel_type_sum = np.sum(rel_attrs, axis=(0, 1))
        assert rel_type_sum[0] == 1
        assert rel_type_sum[1] == 1
        assert rel_type_sum[2] == 1
        assert rel_type_sum[3] == 1
        assert rel_type_sum[4] == 1
        assert rel_type_sum[5] == (N - 2) * 2
        assert rel_type_sum[6] == (N - 3) * 2
        assert rel_type_sum[7] == N - 1

    else:
        raise AssertionError("unsupported env")

    return attrs, states, actions, rel_attrs


def gen_Rope(info):
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, time_step = info['n_rollout'], info['time_step']
    dt, video, args, phase = info['dt'], info['video'], info['args'], info['phase']

    # np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    attr_dim = args.attr_dim  # root, child
    state_dim = args.state_dim  # x, y, xdot, ydot
    action_dim = args.action_dim
    param_dim = args.param_dim + 1 # n_ball, init_x, k, damping, gravity

    act_scale = 1.5
    ret_scale = 1.
    
    # dynamic action
    target_x = 0  # Initial target
    x_bounds = (-100, 100) # x_bounds = (-15, 15)  # Bounds for x
    threshold = 0.08  # Threshold for reaching target
    delta_target = 0.4  # Change in target

    # attr, state, action
    stats = [init_stat(attr_dim), init_stat(state_dim), init_stat(action_dim)]

    engine = RopeEngine(dt, state_dim, action_dim, param_dim)

    group_size = args.group_size
    sub_dataset_size = n_rollout * args.num_workers // args.n_splits
    print('group size', group_size, 'sub_dataset_size', sub_dataset_size)
    assert n_rollout % group_size == 0
    assert args.n_rollout % args.n_splits == 0

    bar = ProgressBar()
    
    print("n_rollout: ",n_rollout)
    print("group_size: ",group_size)
    counter = 0
    for i in bar(range(n_rollout)):
        rollout_idx = thread_idx * n_rollout + i
        group_idx = rollout_idx // group_size
        sub_idx = rollout_idx // sub_dataset_size

        num_obj_range = args.num_obj_range if phase in {'train', 'valid'} else args.extra_num_obj_range
        num_obj = num_obj_range[sub_idx]

        rollout_dir = os.path.join(data_dir, str(rollout_idx))

        param_file = os.path.join(data_dir, str(group_idx) + '.param')

        os.system('mkdir -p ' + rollout_dir)

        if rollout_idx % group_size == 0:
            engine.init(param=(num_obj, None, None, None, None, None))
            torch.save(engine.get_param(), param_file)
            counter += 1
        else:
            while not os.path.isfile(param_file):
                time.sleep(0.5)
            param = torch.load(param_file)
            engine.init(param=param)

        for j in range(time_step):
            states = engine.get_state()
            ############################################################## obj x
            # if j == 0:
            #     init_pos_x = states[0, 0] # first ball, x postion, at 0 timestep
            #     init_pos_y = states[0, 1]
            # states[:, 0] = states[:, 0] - init_pos_x
            # states[:, 1] = states[:, 1] - init_pos_y

            states_ctl = states[0]
            
            # ------- change action generation
            current_x = states_ctl[0]
            # if reaches the target
            count = 0
            if abs(current_x - target_x) < threshold:
                # Choose a new target_x within bounds
                while True:
                    count += 1
                    new_target = target_x + np.random.choice([-delta_target, delta_target])
                    if x_bounds[0] <= new_target <= x_bounds[1]:
                        target_x = new_target
                        break

                    if count >= 10:
                        # print('infinite loop?')
                        target_x = max(min(current_x, x_bounds[1]),x_bounds[0])
                        break
            elif abs(current_x - target_x) > delta_target:
                if target_x > current_x:
                    target_x = current_x + delta_target
                else:
                    target_x = current_x - delta_target
                    
            act_t = np.zeros((engine.num_obj, action_dim))
            act_t[0, 0] = (np.random.rand() * 2 - 1.) * act_scale - (current_x - target_x) * ret_scale
            # ------- end change action generation
            
            
            # act_t[0, 0] = (np.random.rand() * 2 - 1.) * act_scale - states_ctl[0] * ret_scale

            engine.set_action(action=act_t)

            actions = engine.get_action()

            n_obj = engine.num_obj

            pos = states[:, :2].copy()
            vec = states[:, 2:].copy()

            '''reset velocity'''
            if j > 0:
                vec = (pos - states_all[j - 1, :, :2]) / dt

            if j == 0:
                attrs_all = np.zeros((time_step, n_obj, attr_dim))
                states_all = np.zeros((time_step, n_obj, state_dim))
                actions_all = np.zeros((time_step, n_obj, action_dim))

            '''attrs: [1, 0] => root; [0, 1] => child'''
            assert attr_dim == 2
            attrs = np.zeros((n_obj, attr_dim))
            # category: the first ball is fixed
            attrs[0, 0] = 1
            attrs[1:, 1] = 1

            assert np.sum(attrs[:, 0]) == 1
            assert np.sum(attrs[:, 1]) == engine.num_obj - 1

            attrs_all[j] = attrs
            states_all[j, :, :2] = pos
            states_all[j, :, 2:] = vec
            actions_all[j] = actions

            data = [attrs, states_all[j], actions_all[j]]

            store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

            engine.step()
        # import pdb
        # orig_states_all = states_all.copy()
        # print(orig_states_all.shape,states_all.shape)
        # initial_pos =states_all[0, :, :2].copy()
        # for j in range(time_step):
        #     states_all[j, :, :2] -= initial_pos


        datas = [attrs_all.astype(np.float64), states_all.astype(np.float64), actions_all.astype(np.float64)]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0]
            stats[j] = combine_stat(stats[j], stat)

    return stats



class PhysicsDataset(Dataset):

    def __init__(self, args, phase):
        
        self.args = args
        self.phase = phase
        self.data_dir = os.path.join(self.args.dataf, phase)
        print(f'\n\n\n\n Loading data at {self.data_dir}')
        if gethostname().startswith('netmit') and phase == 'extra':
            self.data_dir = self.args.dataf + '_' + phase

        self.stat_path = os.path.join(self.args.dataf, 'stat.h5')
        self.stat = None

        os.system('mkdir -p ' + self.data_dir)

        if args.env in ['Rope', 'Soft', 'Swim']:
            self.data_names = ['attrs', 'states', 'actions']
        else:
            raise AssertionError("Unknown env")

        ratio = self.args.train_valid_ratio
        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase in {'valid', 'extra'}:
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

        self.T = self.args.len_seq

    def load_data(self):
        self.stat = load_data(self.data_names, self.stat_path)

    def gen_data(self):
        # if the data hasn't been generated, generate the data
        n_rollout, time_step, dt = self.n_rollout, self.args.time_step, self.args.dt
        assert n_rollout % self.args.num_workers == 0

        print("Generating data ... n_rollout=%d, time_step=%d" % (n_rollout, time_step))

        infos = []
        for i in range(self.args.num_workers):
            info = {'thread_idx': i,
                    'data_dir': self.data_dir,
                    'data_names': self.data_names,
                    'n_rollout': n_rollout // self.args.num_workers,
                    'time_step': time_step,
                    'dt': dt,
                    'video': False,
                    'phase': self.phase,
                    'args': self.args}

            infos.append(info)

        cores = self.args.num_workers
        pool = mp.Pool(processes=cores)

        env = self.args.env

        if env == 'Rope':
            data = pool.map(gen_Rope, infos)
        elif env == 'Soft':
            data = pool.map(gen_Soft, infos)
        elif env == 'Swim':
            data = pool.map(gen_Swim, infos)
        else:
            raise AssertionError("Unknown env")

        print("Training data generated, warpping up stats ...")

        if self.phase == 'train':
            # states [x, y, angle, xdot, ydot, angledot], action [x, xdot]
            if env in ['Rope', 'Soft', 'Swim']:
                self.stat = [init_stat(self.args.attr_dim),
                             init_stat(self.args.state_dim),
                             init_stat(self.args.action_dim)]

            for i in range(len(data)):
                for j in range(len(self.stat)):
                    self.stat[j] = combine_stat(self.stat[j], data[i][j])

            if self.args.gen_stat:
                print("Storing stat to %s" % self.stat_path)
                store_data(self.data_names, self.stat, self.stat_path)
            else:
                print("stat will be discarded")
        else:
            print("Loading stat from %s ..." % self.stat_path)

            if env in ['Rope', 'Soft', 'Swim']:
                self.stat = load_data(self.data_names, self.stat_path)

    def __len__(self):
        return self.n_rollout * (self.args.time_step - self.T)

    def __getitem__(self, idx):
        idx_rollout = idx // (self.args.time_step - self.T)
        idx_timestep = idx % (self.args.time_step - self.T)

        # prepare input data
        seq_data = None
        for t in range(self.T + 1):
            data_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep + t) + '.h5')
            data = load_data(self.data_names, data_path)
            data = prepare_input(data, self.stat, self.args)
            if seq_data is None:
                seq_data = [[d] for d in data]
            else:
                for i, d in enumerate(data):
                    seq_data[i].append(d)
        seq_data = [np.array(d).astype(np.float32) for d in seq_data]

        return seq_data
