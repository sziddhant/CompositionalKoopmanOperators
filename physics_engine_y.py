import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pymunk
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Polygon
from pymunk.vec2d import Vec2d

from utils import rand_float, rand_int, calc_dis, norm


class Engine(object):
    def __init__(self, dt, state_dim, action_dim, param_dim):
        self.dt = dt
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.param_dim = param_dim

        self.state = None
        self.action = None
        self.param = None

        self.init()

    def init(self):
        pass

    def get_param(self):
        return self.param.copy()

    def set_param(self, param):
        self.param = param.copy()

    def get_state(self):
        return self.state.copy()

    def set_state(self, state):
        self.state = state.copy()

    def get_scene(self):
        return self.state.copy(), self.param.copy()

    def set_scene(self, state, param):
        self.state = state.copy()
        self.param = param.copy()

    def get_action(self):
        return self.action.copy()

    def set_action(self, action):
        self.action = action.copy()

    def d(self, state, t, param):
        # time derivative
        pass

    def step(self):
        pass

    def render(self, state, param):
        pass

    def clean(self):
        pass


class RopeEngine(Engine):

    def __init__(self, dt, state_dim, action_dim, param_dim,
                 num_mass_range=[4, 8], k_range=[500., 1500.], gravity_range=[-2., -8.],
                 position_range=[-10, 10], bihop=True):

        # state_dim = 4
        # action_dim = 1
        # param_dim = 5
        # param [n_ball, init_x, k, damping, gravity]

        self.radius = 0.06
        self.mass = 1.

        self.num_mass_range = num_mass_range
        self.k_range = k_range
        self.gravity_range = gravity_range
        self.position_range = position_range
        
        self.y_range = [-5,5]

        self.bihop = bihop

        super(RopeEngine, self).__init__(dt, state_dim, action_dim, param_dim)

    def init(self, param=None):
        if param is None:
            self.n_ball, self.init_x, self.init_y, self.k, self.damping, self.gravity = [None] * 6
        else:
            self.n_ball, self.init_x, self.init_y, self.k, self.damping, self.gravity = param
            self.n_ball = int(self.n_ball)

        num_mass_range = self.num_mass_range
        position_range = self.position_range
        y_range = self.y_range
        if self.n_ball is None:
            self.n_ball = rand_int(num_mass_range[0], num_mass_range[1])
        if self.init_x is None:
            self.init_x = np.random.rand() * (position_range[1] - position_range[0]) + position_range[0]
        if self.k is None:
            self.k = rand_float(self.k_range[0], self.k_range[1])
        if self.damping is None:
            self.damping = self.k / 20.
        if self.gravity is None:
            self.gravity = rand_float(self.gravity_range[0], self.gravity_range[1])
        if self.init_y is None:
            self.init_y = np.random.rand() * (y_range[1] - y_range[0]) + y_range[0]
        self.param = np.array([self.n_ball, self.init_x, self.init_y, self.k, self.damping, self.gravity])

        # print('Env Rope param: n_ball=%d, init_x=%.4f, k=%.4f, damping=%.4f, gravity=%.4f' % (
        #     self.n_ball, self.init_x, self.k, self.damping, self.gravity))

        self.space = pymunk.Space()
        self.space.gravity = (0., self.gravity)

        self.height = 1.0 # initial y height (could be changed)
        self.rest_len = 0.3

        self.add_masses()
        self.add_rels()

        self.state_prv = None

    @property
    def num_obj(self):
        return self.n_ball

    def add_masses(self):
        inertia = pymunk.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        x = self.init_x
        y = self.init_y
        self.balls = []

        for i in range(self.n_ball):
            body = pymunk.Body(self.mass, inertia)
            body.position = Vec2d(x, y)
            shape = pymunk.Circle(body, self.radius, (0, 0))

            if i == 0:
                # fix the first mass to a specific height
                move_joint = pymunk.GrooveJoint(self.space.static_body, body, (-15, y), (15, y), (0, 0))
                self.space.add(body, shape, move_joint)
            else:
                self.space.add(body, shape)

            self.balls.append(body)
            y -= self.rest_len

    def add_rels(self):
        give = 1. + 0.075
        # add springs over adjacent balls
        for i in range(self.n_ball - 1):
            c = pymunk.DampedSpring(
                self.balls[i], self.balls[i + 1], (0, 0), (0, 0),
                rest_length=self.rest_len * give, stiffness=self.k, damping=self.damping)
            self.space.add(c)

        # add bihop springs
        if self.bihop:
            for i in range(self.n_ball - 2):
                c = pymunk.DampedSpring(
                    self.balls[i], self.balls[i + 2], (0, 0), (0, 0),
                    rest_length=self.rest_len * give * 2, stiffness=self.k * 0.5, damping=self.damping)
                self.space.add(c)

    def add_impulse(self):
        impulse = (self.action[0], 0)
        self.balls[0].apply_impulse_at_local_point(impulse=impulse, point=(0, 0))

    def get_param(self):
        return self.n_ball, self.init_x, self.init_y, self.k, self.damping, self.gravity

    def get_state(self):
        state = np.zeros((self.n_ball, 4))
        for i in range(self.n_ball):
            ball = self.balls[i]
            state[i] = np.array([ball.position[0], ball.position[1], ball.velocity[0], ball.velocity[1]])

        vel_dim = self.state_dim // 2
        if self.state_prv is None:
            state[:, vel_dim:] = 0
        else:
            state[:, vel_dim:] = (state[:, :vel_dim] - self.state_prv[:, :vel_dim]) / self.dt

        return state

    def step(self):
        self.add_impulse()
        self.state_prv = self.get_state()
        self.space.step(self.dt)

    def render(self, states, actions=None, param=None, video=True, image=False, path=None,
               act_scale=None, draw_edge=True, lim=(-20, 20, -20, 20), states_gt=None,
               count_down=False, gt_border=False):
        if video:
            video_path = path + '.avi'
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            print('Save video as %s' % video_path)
            out = cv2.VideoWriter(video_path, fourcc, 25, (640, 480))

        if image:
            image_path = path + '_img'
            print('Save images to %s' % image_path)
            os.system('mkdir -p %s' % image_path)

        c = ['royalblue', 'tomato', 'limegreen', 'orange', 'violet', 'chocolate', 'lightsteelblue']

        time_step = states.shape[0]
        n_ball = states.shape[1]

        if actions is not None and actions.ndim == 3:
            '''get the first ball'''
            actions = actions[:, 0, :]

        for i in range(time_step):
            fig, ax = plt.subplots(1)
            plt.xlim(lim[0], lim[1])
            plt.ylim(lim[2], lim[3])
            plt.axis('off')

            if draw_edge:
                cnt = 0
                for x in range(n_ball - 1):
                    plt.plot([states[i, x, 0], states[i, x + 1, 0]],
                             [states[i, x, 1], states[i, x + 1, 1]],
                             '-', color=c[1], lw=2, alpha=0.5)

            circles = []
            circles_color = []
            for j in range(n_ball):
                circle = Circle((states[i, j, 0], states[i, j, 1]), radius=self.radius * 5 / 4)
                circles.append(circle)
                circles_color.append(c[0])

            pc = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=1.)
            ax.add_collection(pc)

            if states_gt is not None:
                circles = []
                circles_color = []
                for j in range(n_ball):
                    circle = Circle((states_gt[i, j, 0], states_gt[i, j, 1]), radius=self.radius * 5 / 4)
                    circles.append(circle)
                    circles_color.append('limegreen')
                pc = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=1.)
                ax.add_collection(pc)

            if actions is not None:
                F = actions[i, 0] / 4
                normF = norm(F)
                if normF < 1e-10:
                    pass
                else:
                    ax.arrow(states[i, 0, 0] + F / normF * 0.1, states[i, 0, 1],
                             F, 0., fc='Orange', ec='Orange', width=0.04, head_width=0.2, head_length=0.2)

            ax.set_aspect('equal')

            font = {'family': 'serif',
                    'color': 'darkred',
                    'weight': 'normal',
                    'size': 16}
            if count_down:
                plt.text(-2.5, 1.5, 'CountDown: %d' % (time_step - i - 1), fontdict=font)

            plt.tight_layout()

            if video:
                fig.canvas.draw()
                frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
                if i == time_step - 1:
                    for _ in range(5):
                        out.write(frame)

            if image:
                plt.savefig(os.path.join(image_path, 'fig_%s.png' % i), bbox_inches='tight')

            plt.close()

        if video:
            out.release()


# ===================================================================
'''
For Soft and Swim
'''


def get_init_p_fish_8():
    init_p = np.zeros((8, 3))
    init_p[0, :] = np.array([0, 0, 2])
    init_p[1, :] = np.array([0, 1, 0])
    init_p[2, :] = np.array([0, 2, 2])
    init_p[3, :] = np.array([0, 3, 0])
    init_p[4, :] = np.array([1, 0, 2])
    init_p[5, :] = np.array([1, 1, 0])
    init_p[6, :] = np.array([1, 2, 2])
    init_p[7, :] = np.array([1, 3, 0])
    return init_p


def sample_init_p_flight(n_box, shape_type=None, aug=False, train=False,
                         min_offset=False, max_offset=False):
    assert 5 <= n_box < 10
    c_box_dict = {
        5: [[1, 3, 1], [2, 1, 2]],
        6: [[3, 3], [2, 2, 2]],
        7: [[2, 3, 2], [1, 2, 1, 2, 1], [2, 1, 1, 1, 2]],
        8: [[2, 2, 2, 2], [1, 2, 2, 2, 1], [2, 1, 2, 1, 2], [3, 2, 3]],
        9: [[2, 2, 1, 2, 2], [1, 2, 3, 2, 1], [2, 1, 3, 1, 2], [3, 3, 3]],
    }

    if shape_type is None:
        shape_type = rand_int(0, len(c_box_dict[n_box]))
    else:
        shape_type = shape_type % len(c_box_dict[n_box])

    c_box = c_box_dict[n_box][shape_type]

    init_p = np.zeros((n_box, 3))
    y_offset = np.zeros(len(c_box))

    for i in range(1, (len(c_box) + 1) // 2):
        left = c_box[i - 1]
        right = c_box[i]
        y_offset[i] = rand_int(1 - right, left)
        if min_offset: y_offset[i] = 1 - right
        if max_offset: y_offset[i] = left
        y_offset[len(c_box) - i] = - y_offset[i]
        assert len(c_box) - i > i

    y = np.zeros(len(c_box))
    for i in range(1, len(c_box)):
        y[i] = y[i - 1] + y_offset[i]
    y -= y.min()

    # print('y_offset', y_offset, 'y', y)

    while True:
        idx = 0
        for i, c in enumerate(c_box):
            for j in range(c):
                # if not train:
                if False:
                    material = 2 if j < c - 1 or c == 1 else 0
                else:
                    r = np.random.rand()
                    if c == 1:
                        r_actuated, r_soft, r_rigid = 0.25, 0.25, 0.5
                    elif j == 0:
                        r_actuated, r_soft, r_rigid = 0.0, 0.5, 0.5
                    elif j == c - 1:
                        r_actuated, r_soft, r_rigid = 0.75, 0.25, 0.0
                    else:
                        r_actuated, r_soft, r_rigid = 0.4, 0.2, 0.4
                    if r < r_actuated:
                        material = 0
                    elif r < r_actuated + r_soft:
                        material = 1
                    else:
                        material = 2
                init_p[idx, :] = np.array([i, y[i] + j, material])
                idx += 1

        if (init_p[:, 2] == 0).sum() >= 2:
            break

    # print('init_p', init_p)

    if aug:
        if np.random.rand() > 0.5:
            '''flip y'''
            init_p[:, 1] = -init_p[:, 1]
        if np.random.rand() > 0.5:
            '''flip x'''
            init_p[:, 0] = -init_p[:, 0]
        if np.random.rand() > 0.5:
            '''swap x and y'''
            x, y = init_p[:, 0], init_p[:, 1]
            init_p[:, 0], init_p[:, 1] = y.copy(), x.copy()

    # print('init_p', init_p)

    return init_p


def sample_init_p_regular(n_box, shape_type=None, aug=False):
    print('sample_init_p')
    init_p = np.zeros((n_box, 3))

    if shape_type is None: shape_type = rand_int(0, 4)
    print('shape_type', shape_type)

    if shape_type == 0:  # 0 or u shape
        init_p[0, :] = np.array([0, 0, 2])
        init_p[1, :] = np.array([-1, 0, 2])
        init_p[2, :] = np.array([1, 0, 2])
        idx = 3
        y = 0
        x = [-1, 0, 1]
        res = n_box - 3
        while res > 0:
            y += 1
            if res == 3:
                i_list = [0, 1, 2]
            else:
                i_list = [0, 2]
            material = [0, 1][int(np.random.rand() < 0.5 and res > 3)]
            for i in i_list:
                init_p[idx, :] = np.array([x[i], y, material])
                idx += 1
                res -= 1

    elif shape_type == 1:  # 1 shape
        init_p[0, :] = np.array([0, 0, 2])
        for i in range(1, n_box):
            material = [0, 1][int(np.random.rand() < 0.5 and i < n_box - 1)]
            init_p[i, :] = np.array([0, i, material])

    elif shape_type == 2:  # I shape
        if n_box < 7:
            init_p[0, :] = np.array([0, 0, 2])
            for i in range(1, n_box - 3):
                material = [0, 1][int(np.random.rand() < 0.5 and i < n_box - 1)]
                init_p[i, :] = np.array([0, i, material])
            init_p[n_box - 1, :] = np.array([-1, n_box - 3, 0])
            init_p[n_box - 2, :] = np.array([0, n_box - 3, 0])
            init_p[n_box - 3, :] = np.array([1, n_box - 3, 0])
        else:
            init_p[0, :] = np.array([-1, 0, 2])
            init_p[1, :] = np.array([0, 0, 2])
            init_p[2, :] = np.array([1, 0, 2])
            for i in range(3, n_box - 3):
                material = [0, 1][int(np.random.rand() < 0.5 and i < n_box - 1)]
                init_p[i, :] = np.array([0, i - 2, material])
            init_p[n_box - 1, :] = np.array([-1, n_box - 5, 0])
            init_p[n_box - 2, :] = np.array([0, n_box - 5, 0])
            init_p[n_box - 3, :] = np.array([1, n_box - 5, 0])

    elif shape_type == 3:  # T shape
        if n_box < 6:
            init_p[0, :] = np.array([-1, 0, 2])
            init_p[1, :] = np.array([0, 0, 2])
            init_p[2, :] = np.array([1, 0, 2])
            for i in range(3, n_box):
                material = [0, 1][int(np.random.rand() < 0.5 and i < n_box - 1)]
                init_p[i, :] = np.array([0, i - 2, material])
        else:
            init_p[0, :] = np.array([-2, 0, 2])
            init_p[1, :] = np.array([-1, 0, 2])
            init_p[2, :] = np.array([0, 0, 2])
            init_p[3, :] = np.array([1, 0, 2])
            init_p[4, :] = np.array([2, 0, 2])
            for i in range(5, n_box):
                material = [0, 1][int(np.random.rand() < 0.5 and i < n_box - 1)]
                init_p[i, :] = np.array([0, i - 4, material])

    elif shape_type == 4:  # stronger T
        assert n_box == 10
        init_p[0, :] = np.array([0, -4, 0])
        init_p[1, :] = np.array([1, -4, 1])
        init_p[2, :] = np.array([0, -3, 0])
        init_p[3, :] = np.array([1, -3, 0])
        init_p[4, :] = np.array([0, -2, 1])
        init_p[5, :] = np.array([1, -2, 0])
        init_p[6, :] = np.array([-1, -1, 2])
        init_p[7, :] = np.array([0, -1, 2])
        init_p[8, :] = np.array([1, -1, 2])
        init_p[9, :] = np.array([2, -1, 2])

    if aug:
        if np.random.rand() > 0.5:
            '''flip y'''
            init_p[:, 1] = -init_p[:, 1]
        if np.random.rand() > 0.5:
            '''swap x and y'''
            x, y = init_p[:, 0], init_p[:, 1]
            init_p[:, 0], init_p[:, 1] = y.copy(), x.copy()

    return init_p
