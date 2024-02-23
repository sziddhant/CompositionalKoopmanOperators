import os

from config import gen_args
from data import normalize, denormalize
from models.CompositionalKoopmanOperators import CompositionalKoopmanOperators
from models.KoopmanBaselineModel import KoopmanBaseline
# from physics_engine import SoftEngine, RopeEngine, SwimEngine
from physics_engine_y import RopeEngine
from utils import *
from utils import to_var, to_np, Tee
from progressbar import ProgressBar
import time

# 135 is baseline, 140 is obj
args = gen_args()
print_args(args)
'''
args.fit_num is # of trajectories used for SysID
'''

data_names = ['attrs', 'states', 'actions']
prepared_names = ['attrs', 'states', 'actions', 'rel_attrs']
# data_dir = os.path.join(args.dataf, args.eval_set)
# args.stat_path = "data/data_baseline_Rope/stat.h5"
# data_dir = "data/data_baseline_Rope/train"
# print(f"Load stored dataset statistics from {args.stat_path} and {data_dir}!")
# stat = load_data(data_names, args.stat_path)

if args.env == 'Rope':
    engine = RopeEngine(args.dt, args.state_dim, args.action_dim, args.param_dim)
else:
    assert False


os.system('mkdir -p ' + args.evalf)
log_path = os.path.join(args.evalf, 'log.txt')
tee = Tee(log_path, 'w')

'''
model
'''
# build model
use_gpu = torch.cuda.is_available()
if not args.baseline:
    """ Koopman model for Baseline"""
    model_base = CompositionalKoopmanOperators(args, residual=False, use_gpu=use_gpu)

    model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (85, 0))
    
    print("Loading saved checkpoint from %s" % model_path)
    device = torch.device('cuda:0') if use_gpu else torch.device('cpu')
    model_base.load_state_dict(torch.load(model_path,map_location=device))
    model_base.eval()
    if use_gpu: model_base.cuda()
    
    model_obj = CompositionalKoopmanOperators(args, residual=False, use_gpu=use_gpu)

    # load pretrained checkpoint

    model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (80, 0))
    device = torch.device('cuda:0') if use_gpu else torch.device('cpu')
    model_obj.load_state_dict(torch.load(model_path,map_location=device))
    model_obj.eval()
    if use_gpu: model_obj.cuda()

else:
    """ Koopman Baselinese """
    model = KoopmanBaseline(args)

'''
eval
'''


def get_more_trajectories(roll_idx,baseline=True):
    if baseline:
        data_dir = os.path.join("data/data_baseline_Rope", args.eval_set)
    else:
        data_dir = os.path.join("data/data_obj_Rope", args.eval_set)
        
    group_idx = roll_idx // args.group_size
    offset = group_idx * args.group_size

    all_seq = [[], [], [], []]

    for i in range(1, args.fit_num + 1):
        new_idx = (roll_idx + i - offset) % args.group_size + offset
        seq_data = load_data(prepared_names, os.path.join(data_dir, str(new_idx) + '.rollout.h5'))
        for j in range(4):
            all_seq[j].append(seq_data[j])

    all_seq = [np.array(all_seq[j], dtype=np.float32) for j in range(4)]
    return all_seq

def eval_data(model, idx_rollout, baseline=True):
    print(f'\n=== Forward Simulation on Example {roll_idx} ===')
    if baseline:
        data_dir = os.path.join("data/data_baseline_Rope", args.eval_set)
        print(os.path.join(data_dir, str(idx_rollout) + '.rollout.h5'))
        seq_data = load_data(prepared_names, os.path.join(data_dir, str(idx_rollout) + '.rollout.h5'))
        stat_path = "data/data_baseline_Rope/stat.h5"
        stat = load_data(data_names, stat_path)
    else:
        data_dir = os.path.join("data/data_obj_Rope", args.eval_set)
        print(os.path.join(data_dir, str(idx_rollout) + '.rollout.h5'))
        seq_data = load_data(prepared_names, os.path.join(data_dir, str(idx_rollout) + '.rollout.h5'))
        stat_path = "data/data_obj_Rope/stat.h5"
        stat = load_data(data_names, stat_path)
    attrs, states, actions, rel_attrs = [to_var(d.copy(), use_gpu=use_gpu) for d in seq_data]

    seq_data = denormalize(seq_data, stat)
    attrs_gt, states_gt, action_gt = seq_data[:3]

    param_file = os.path.join(data_dir, str(idx_rollout // args.group_size) + '.param')
    param = torch.load(param_file)
    engine.init(param)

    '''
    fit data
    '''
    fit_data = get_more_trajectories(roll_idx, baseline)
    fit_data = [to_var(d, use_gpu=use_gpu) for d in fit_data]
    bs = args.fit_num

    ''' T x N x D (denormalized)'''
    states_pred = states_gt.copy()
    states_pred[1:] = 0

    ''' T x N x D (normalized)'''
    s_pred = states.clone()

    '''
    reconstruct loss
    '''
    attrs_flat = get_flat(fit_data[0])
    states_flat = get_flat(fit_data[1])
    actions_flat = get_flat(fit_data[2])
    rel_attrs_flat = get_flat(fit_data[3])

    g = model.to_g(attrs_flat, states_flat, rel_attrs_flat, args.pstep)
    g = g.view(torch.Size([bs, args.time_step]) + g.size()[1:])

    G_tilde = g[:, :-1]
    H_tilde = g[:, 1:]
    U_tilde = fit_data[2][:, :-1]

    G_tilde = get_flat(G_tilde, keep_dim=True)
    H_tilde = get_flat(H_tilde, keep_dim=True)
    U_tilde = get_flat(U_tilde, keep_dim=True)

    _t = time.time()
    A, B, fit_err = model.system_identify(
        G=G_tilde, H=H_tilde, U=U_tilde, rel_attrs=fit_data[3][:1, 0], I_factor=args.I_factor)
    _t = time.time() - _t

    '''
    predict
    '''

    g = model.to_g(attrs, states, rel_attrs, args.pstep)

    pred_g = None
    for step in range(0, args.time_step - 1):
        # prepare input data

        if step == 0:
            current_s = states[step:step + 1]
            current_g = g[step:step + 1]
            states_pred[step] = states_gt[step]
        else:
            '''current state'''
            if args.eval_type == 'valid':
                current_s = states[step:step + 1]
            elif args.eval_type == 'rollout':
                current_s = s_pred[step:step + 1]

            '''current g'''
            if args.eval_type in {'valid', 'rollout'}:
                current_g = model.to_g(attrs[step:step + 1], current_s, rel_attrs[step:step + 1], args.pstep)
            elif args.eval_type == 'koopman':
                current_g = pred_g

        '''next g'''
        pred_g = model.step(g=current_g, u=actions[step:step + 1], rel_attrs=rel_attrs[step:step + 1])

        '''decode s'''
        pred_s = model.to_s(attrs=attrs[step:step + 1], gcodes=pred_g,
                            rel_attrs=rel_attrs[step:step + 1], pstep=args.pstep)

        pred_s_np_denorm = denormalize([to_np(pred_s)], [stat[1]])[0]

        states_pred[step + 1:step + 2] = pred_s_np_denorm
        d = args.state_dim // 2
        states_pred[step + 1:step + 2, :, :d] = states_pred[step:step + 1, :, :d] + \
                                                args.dt * states_pred[step + 1:step + 2, :, d:]

        s_pred_next = normalize([states_pred[step + 1:step + 2]], [stat[1]])[0]
        s_pred[step + 1:step + 2] = to_var(s_pred_next, use_gpu=use_gpu)
        
    return states_pred
def eval(idx_rollout, video=True):
    print(f'\n=== Forward Simulation on Example {roll_idx} ===')
    
    data_dir = os.path.join("data/data_baseline_Rope", args.eval_set)
    print(os.path.join(data_dir, str(idx_rollout) + '.rollout.h5'))
    seq_data = load_data(prepared_names, os.path.join(data_dir, str(idx_rollout) + '.rollout.h5'))
    stat_path = "data/data_baseline_Rope/stat.h5"
    stat = load_data(data_names, stat_path)
    
    initial = load_data(prepared_names, os.path.join(data_dir, str(idx_rollout), "0.h5"))
    initial_pos = initial[1][0][:2] # 1 is state 's idx, 0 is the first ball, :2 is x and y position

    seq_data = denormalize(seq_data, stat)
    attrs_gt, states_gt, action_gt = seq_data[:3]

    data_dir = os.path.join("data/data_obj_Rope", args.eval_set)
    param_file = os.path.join(data_dir, str(idx_rollout // args.group_size) + '.param')
    param = torch.load(param_file)
    engine.init(param)

    '''
    fit data
    '''
    # fit_data = get_more_trajectories(roll_idx,basel)
    # fit_data = [to_var(d, use_gpu=use_gpu) for d in fit_data]
    # bs = args.fit_num

    ''' T x N x D (denormalized)'''
    states_pred_baseline = eval_data(model_base, idx_rollout, 1)
    states_pred_obj = eval_data(model_obj, idx_rollout, 0)


    for timesteps in states_pred_obj:
        for obj in timesteps:
            obj[0] += initial_pos[0]
            obj[1] += initial_pos[1]


    if video:
        save_path = "pred_states" + args.obj + ".txt"
        with open(save_path, 'a') as f:
            f.write("-----------------------------------------------------------------------\n")
            for sublist in states_pred_obj:
                line = '\n'.join(map(str, sublist))
                f.write(f"{line}\n")
                f.write("\n")
        # engine.render(states_pred, seq_data[2], param, act_scale=args.act_scale, video=True, image=True,
        #               path=os.path.join(args.evalf, str(idx_rollout) + '.pred'),
        #               states_gt=states_gt)
        engine.render_cmp(states=states_pred_baseline, states_obj=states_pred_obj,video=True,
                      path=os.path.join(args.evalf, str(idx_rollout) + '.pred'),
                      states_gt=states_gt)
        # print(states_gt.shape)
        # for t in states_gt:
        #     print(t)
        #     break

if __name__ == '__main__':

    num_train = int(args.n_rollout * args.train_valid_ratio)
    num_valid = args.n_rollout - num_train

    ls_rollout_idx = np.arange(0, num_valid, num_valid // args.n_splits)

    if args.demo:
        ls_rollout_idx = np.arange(8) * 25

    for roll_idx in ls_rollout_idx[:3]:
        eval(roll_idx)
