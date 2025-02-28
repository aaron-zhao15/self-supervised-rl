import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
import math

import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import cv2
import pickle

class ContinuousMovableObject:
    def __init__(self, x, y, grid_size, speed=1, occupancy_radius=1):
        self.position = np.array([x, y], dtype=float)
        self.grid_size = grid_size
        self.occupancy_radius = occupancy_radius
        self.speed = speed

    def move(self, movement, other_agents=None):
        new_position = self.position + self.speed * np.array(movement)
        new_position = np.clip(new_position, 0, self.grid_size - 1)
        
        for other_agent in other_agents:
            distance = np.linalg.norm(new_position - other_agent.position)
            if distance < self.occupancy_radius + other_agent.occupancy_radius:
                return  # Collision detected, cancel the move

        self.position = new_position

class PirateEnv(gym.Env):
    def __init__(self, num_agents=3, grid_size=20, disable_distance=3, occupancy_radius=1):
        super(PirateEnv, self).__init__()

        self.num_agents = num_agents
        self.n = num_agents
        self.num_targets = num_agents
        self.grid_size = grid_size
        self.disable_distance = disable_distance
        self.max_steps = 100
        self.steps = 0
        self.occupancy_radius = occupancy_radius
        self.capture_limit = 10

        self.action_space = [spaces.Tuple((
            spaces.Discrete(2),  # Action type: 0 (move), 1 (capture)
            spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        )) for _ in range(self.num_agents)]

        self.observation_space = [spaces.Box(
            low=0,
            high=self.grid_size,
            shape=(2 * self.num_agents + 3,),
            dtype=np.float32
        ) for _ in range(self.num_agents)]

        self.reset()

    def reset(self):
        self.agents = [ContinuousMovableObject(
            np.random.uniform(0, self.grid_size),
            np.random.uniform(0, self.grid_size),
            self.grid_size,
            speed=0.5 * i + 0.5,
            occupancy_radius=self.occupancy_radius
        ) for i in range(self.num_agents)]

        self.targets = [ContinuousMovableObject(
            np.random.uniform(0, self.grid_size),
            np.random.uniform(0, self.grid_size),
            self.grid_size
        ) for _ in range(self.num_targets)]

        self.targets_disabled = [False] * self.num_targets
        self.agent_capture_count = [0] * self.num_agents
        self.steps = 0

        return self.observations()

    def step(self, actions):
        self.steps += 1
        rewards = np.zeros(self.num_agents)
        done = np.zeros(self.num_agents)
        for i, action in enumerate(actions):
            moving = action[0]
            capturing = action[3]
            if moving:  # Move action
                other_agents = [agent for j, agent in enumerate(self.agents) if j != i]
                theta = action[1]
                movement = np.array([np.cos(theta), np.sin(theta)])
                self.agents[i].move(movement, other_agents)
            
            if capturing and self.agent_capture_count[i] < self.capture_limit:
                target = self.targets[i]
                distance = np.linalg.norm(self.agents[i].position - target.position)
                if distance <= self.disable_distance and not self.targets_disabled[i]:
                    rewards[i] = 10  # Capture reward
                    self.targets_disabled[i] = True
                    done[i] = True
                else:
                    rewards[i] = 0  # Failed capture penalty
                self.agent_capture_count[i] += 1

            if self.agent_capture_count[i] >= self.capture_limit:
                done[i] = True

        for i, target in enumerate(self.targets):
            if not self.targets_disabled[i]:
                target.move(np.random.uniform(-1, 1, size=2), self.agents)

        move_rewards = self.dist_rewards()
        rewards += move_rewards

        if self.steps >= self.max_steps:
            done = np.ones_like(done)
        observations = self.observations()

        return observations, rewards, done.tolist(), {}

    def dist_rewards(self):
        return np.array([-np.linalg.norm(agent.position - target.position) for agent, target in zip(self.agents, self.targets)])

    def observations(self):
        obs = [[] for _ in self.agents]
        for agent_id, agent in enumerate(self.agents):
            obs[agent_id].extend(agent.position)
            for other_agent_id, other_agent in enumerate(self.agents):
                if agent_id != other_agent_id:
                    obs[agent_id].extend(other_agent.position)
            obs[agent_id].extend(self.targets[agent_id].position)
            obs[agent_id].extend([self.targets_disabled[agent_id]])
        return np.array(obs)
    
    def render(self, mode='human'):
        canvas_size = 500
        scale = canvas_size / self.grid_size
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255  # White background

        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Vertical lines
            start_point = (int(i * scale), 0)
            end_point = (int(i * scale), canvas_size)
            cv2.line(canvas, start_point, end_point, (200, 200, 200), 1)  # Light gray grid lines
            # Horizontal lines
            start_point = (0, int(i * scale))
            end_point = (canvas_size, int(i * scale))
            cv2.line(canvas, start_point, end_point, (200, 200, 200), 1)  # Light gray grid lines

        # Draw agents with continuous positions
        for i, agent in enumerate(self.agents):
            color = (0, 255, 0) if self.agent_launched[i] else (255, 0, 0)
            # Scale continuous positions for rendering
            agent_pos = np.clip(agent.position, 0, self.grid_size - 1) * scale
            center = tuple(agent_pos.astype(int))
            cv2.circle(canvas, center, int(scale // 3), color, -1)

        # Draw the target with continuous position
        for j, target in enumerate(self.targets):
            target_color = (0, 0, 255) if not self.targets_disabled[j] else (128, 128, 128)
            # Scale continuous positions for rendering
            target_pos = np.clip(target.position, 0, self.grid_size - 1) * scale
            target_center = tuple(target_pos.astype(int))
            cv2.circle(canvas, target_center, int(scale // 3), target_color, -1)

        return canvas
    
    def save_state(self):
        """
        Returns the current state as a dictionary.
        """
        state = {
            'agents': [(agent.position, agent.speed) for agent in self.agents],
            'targets': [(target.position, target.speed) for target in self.targets],
            'targets_disabled': self.targets_disabled,
            'agent_launched': self.agent_launched,
            'steps': self.steps
        }
        return state

    def load_state(self, state):
        """
        Loads the state from a dictionary.
        """
        self.agents = [
            ContinuousMovableObject(pos[0], pos[1], self.grid_size, speed=speed)
            for pos, speed in state['agents']
        ]
        self.targets = [
            ContinuousMovableObject(pos[0], pos[1], self.grid_size, speed=speed)
            for pos, speed in state['targets']
        ]
        self.targets_disabled = state['targets_disabled']
        self.agent_launched = state['agent_launched']
        self.steps = state['steps']


    def close(self):
        cv2.destroyAllWindows()

def run_environment(env, num_steps=20):
    for _ in range(num_steps):
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        env.step(actions)
        env.render()

def policy_to_action(policies):
    env_actions = []
    for policy in policies:
        action_type = np.argmax(policy[0:2])
        movement = np.array(policy[2:4])
        action = [action_type, movement]
        env_actions.append(action)
    return env_actions

from matplotlib.animation import FuncAnimation
def run_and_visualize(env, agents, num_steps=50, eval_mode="debug", policy_converter=lambda x:x):
    frames = []

    # Initialize state and pre-allocate frame storage if possible
    state = env.reset()

    # Step through the environment and collect frames
    for _ in range(num_steps):
        # Optimize action computation
        if eval_mode == "hybrid":
            actions = []
            for i, (move_agent, decision_agent) in enumerate(agents):
                move_action = move_agent.act(state[i])
                decision_action = decision_agent.act(state[i])
                
                action = np.concatenate((decision_action, move_action))
                actions.append(action)
            actions = policy_converter(actions)
        
        state, rewards, done, _ = env.step(actions)

        # Collect frame only if rendering is enabled
        frame = env.render()
        frames.append(frame)
        if done:
            break

    # Create a video and visualize it in the Jupyter notebook cell
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.axis('off')  # Turn off axis for better visualization
    ax_img = ax.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))  # Display the first frame immediately

    def update_frame(i):
        ax_img.set_data(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        return [ax_img]

    # Use FuncAnimation to create an animation with a lower interval
    anim = FuncAnimation(fig, update_frame, frames=len(frames), interval=100, repeat=False)
    plt.close(fig)  # Prevent duplicate static display of the plot



# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = False
        # self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        self.flag = 0
        self.direction = 0
        self.accel = 0
        # hybrid action space
        self.n_action = 3
        self.hybrid_action_space = True
        # self.hybrid_action_space = False
        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,),
                                            dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def get_action_motions(self, n_actions):
        shape = (2 ** n_actions, 2)
        # print("shape",shape,shape[0])
        motions = np.zeros(shape)
        for idx in range(shape[0]):
            action = binaryEncoding(idx, n_actions)
            motions[idx] = np.dot(action, self.movement)
        max_dist = np.max(np.linalg.norm(motions, ord=2, axis=-1))
        motions /= max_dist
        return motions

    def get_movements(self, n_actions):
        """
        Divides 360 degrees into n_actions and
        assigns how much it should make the agent move in both x,y directions

        usage:  delta_x, delta_y = np.dot(action, movement)
        :param n_actions:
        :return: x,y direction movements for each of the n_actions
        """
        x = np.linspace(0, 2 * np.pi, n_actions + 1)  # 创建等差数列
        # print("x",x)
        y = np.linspace(0, 2 * np.pi, n_actions + 1)
        motion_x = np.around(np.cos(x)[:-1], decimals=3)  # 返回四舍五入后的值
        # print("motion_x",motion_x)
        motion_y = np.around(np.sin(y)[:-1], decimals=3)
        movement = np.vstack((motion_x, motion_y)).T  # 按垂直方向（行顺序）堆叠数组构成一个新的数组

        return movement

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            # self._set_action1(action_n[i], agent, self.action_space[i])
            self._set_action1(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def _set_action1(self, action, agent, action_space, time=None):

        # agent.action.u = np.zeros(self.world.dim_p + 2)  #维度加2 前四维是移动的连续动作参数，后面一个离散动作是控制移动，一个是控制具体操作（抓、踢等）
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if action[0][0] == 2:
            agent.action.u = np.zeros(self.world.dim_p + 3)
        elif action[0][0] == 4:
            agent.action.u = np.zeros(self.world.dim_p + 3)
        elif action[0][0] == 5:
            agent.action.u = np.zeros(self.world.dim_p + 5)
        elif action[0][0] == 6:
            agent.action.u = np.zeros(self.world.dim_p + 4)
        else:
            agent.action.u = np.zeros(self.world.dim_p + 2)

        if agent.movable:
            # physical action
            if self.hybrid_action_space:

                # 4维连续动作参数(移动，停止)
                if action[0][0] == 0:
                    agent.action.u[2] = action[0][5]
                    if action[0][5] == 0:
                        agent.action.u[0] += 0
                        agent.action.u[1] += 0
                    else:
                        agent.action.u[0] += (action[0][1] - action[0][2]) * 2.0
                        agent.action.u[1] += (action[0][3] - action[0][4]) * 2.0
                    agent.action.u[3] = action[0][6]

                # 4个离散动作上下左右，4维连续动作参数[]
                if action[0][0] == 5:

                    if action[0][5] == 1: agent.action.u[0] = -action[0][1] * 2.0
                    if action[0][6] == 1: agent.action.u[0] = action[0][2] * 2.0
                    if action[0][7] == 1: agent.action.u[1] = -action[0][3] * 2.0
                    if action[0][8] == 1: agent.action.u[1] = action[0][4] * 2.0

                    agent.action.u[2] = action[0][5]
                    agent.action.u[3] = action[0][6]
                    agent.action.u[4] = action[0][7]
                    agent.action.u[5] = action[0][8]


                # 4个离散动作上下左右，4维连续动作参数[]
                if action[0][0] == 6:

                    if action[0][5] == 1: agent.action.u[0] = -action[0][1] * 2.0
                    if action[0][6] == 1: agent.action.u[0] = action[0][2] * 2.0
                    if action[0][7] == 1: agent.action.u[1] = -action[0][3] * 2.0
                    if action[0][8] == 1: agent.action.u[1] = action[0][4] * 2.0

                    agent.action.u[2] = action[0][5]
                    agent.action.u[3] = action[0][6]
                    agent.action.u[4] = action[0][7]
                    agent.action.u[5] = action[0][8]

                #  1是连续动作参数 2是离散动作，3是动作的维度
                if action[0][0] == 7:
                    # print("action[0][3]",action[0][3])
                    action_dim = int(action[0][3])
                    # print("action_dim",action_dim)
                    self.movement = self.get_movements(action_dim)
                    self.motions = self.get_action_motions(action_dim)
                    # print("action",int(action[0][2]))
                    # print("self.motions",self.motions,self.motions[int(action[0][2])])
                    action_true = self.motions[int(action[0][2])]
                    # print("action_true", action_true)
                    agent.action.u[0] += action_true[0] * action[0][1] * 2.0
                    agent.action.u[1] += action_true[1] * action[0][1] * 2.0


                #  1是离散动作，2是动作的维度  3是连续动作参数
                if action[0][0] == 8:
                    action_dim = int(action[0][2])
                    self.movement = self.get_movements(action_dim)
                    self.motions = self.get_action_motions(action_dim)
                    action_true = self.motions[int(action[0][1])]
                    accel=action[0][3][int(action[0][1])]
                    agent.action.u[0] += action_true[0] * accel * 2.0
                    agent.action.u[1] += action_true[1] * accel * 2.0


                # direction
                # (空，角度，移动，抓取)
                if action[0][0] == 1:
                    if action[0][2] == 0:
                        agent.action.u[0] += 0
                        agent.action.u[1] += 0
                    else:
                        agent.action.u[0] += np.sin(action[0][1]) * 2.0
                        agent.action.u[1] += np.cos(action[0][1]) * 2.0
                    agent.action.u[2] = action[0][2]
                    agent.action.u[3] = action[0][3]

                ##move
                # （空，角度，力度，调整角度，调整力度，停止）
                if action[0][0] == 2:
                    if action[0][5] == 1:
                        agent.action.u[0] += 0
                        agent.action.u[1] += 0
                    elif action[0][3] == 1:
                        self.direction = action[0][1]
                        agent.action.u[0] += np.sin(self.direction) * self.accel * 2.0
                        agent.action.u[1] += np.cos(self.direction) * self.accel * 2.0
                    elif action[0][4] == 1:
                        self.accel = action[0][2]
                        agent.action.u[0] += np.sin(self.direction) * self.accel * 2.0
                        agent.action.u[1] += np.cos(self.direction) * self.accel * 2.0

                    agent.action.u[2] = action[0][3]
                    agent.action.u[3] = action[0][4]
                    agent.action.u[4] = action[0][5]

                # direction
                # (空，角度，移动，抓取) 对direction进行了修改
                if action[0][0] == 3:
                    if action[0][2] == 0:
                        agent.action.u[0] += 0
                        agent.action.u[1] += 0
                    else:
                        agent.action.u[0] += np.sin(action[0][1]) * 0.5
                        agent.action.u[1] += np.cos(action[0][1]) * 0.5
                    agent.action.u[2] = action[0][2]
                    agent.action.u[3] = action[0][3]

                ##move_hard
                # （空，角度，力度，调整角度，调整力度，停止）
                if action[0][0] == 4:
                    if action[0][5] == 1:
                        agent.action.u[0] += 0
                        agent.action.u[1] += 0
                    elif action[0][3] == 1:
                        self.direction = action[0][1]
                        # agent.action.u[0] += np.sin(self.direction) * self.accel* 2.0
                        # agent.action.u[1] += np.cos(self.direction) * self.accel* 2.0
                    elif action[0][4] == 1:
                        self.accel = action[0][2]
                        agent.action.u[0] += np.sin(self.direction) * self.accel * 2.0
                        agent.action.u[1] += np.cos(self.direction) * self.accel * 2.0

                    agent.action.u[2] = action[0][3]
                    agent.action.u[3] = action[0][4]
                    agent.action.u[4] = action[0][5]

            # print("agent_u",agent.action.u,self.direction)
            sensitivity = 1.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i + env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n


def binaryEncoding(num, size):
    binary = np.zeros(size)
    i = -1
    while num > 0:
        binary[i] = num % 2
        num = num // 2
        i -= 1
    return binary
