import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

#from griddy_render import *
from gym.envs.classic_control import rendering

class SnakyEnv(gym.Env):
    """
    Description:
        A grid world where you have to reach the goal
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: MultiDiscrete((4, 4), 4)
    Actions:
        Type: Discrete(4)
        Num	Action
        0	Move to the left
        1	Move to the right
        2	Move to the north
        3	Move to the south
    Reward:
        Reward is 0 for every step taken and 1 when goal is reached
    Starting State:
        Agent starts in random position and goal is always bottom right
    Episode Termination:
        Agent position is equal to goal position
        Solved Requirements
        Solved fast
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, size=4):
        self.n_squares_height = size
        self.n_squares_width = size

        self.OBJECT_TO_IDX = {
            'goal':1,
            'wall':2,
            'other_agents':3,
            'my_agent':4
        }

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiBinary((len(self.OBJECT_TO_IDX), self.n_squares_height, self.n_squares_width))

        self.seed()
        self.viewer = None
        self.state = None
        
        self.steps_beyond_done = None

        self.players = []
        self.players_in_game = []
        self.agent_cols = []

    def get_n_players(self):
        return len(self.players)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        state = np.full((len(self.OBJECT_TO_IDX)-1, self.n_squares_height, self.n_squares_width), 0)
        #add all agents in random_positions
        all_agents_pos = np.random.choice(range(self.n_squares_height*self.n_squares_width-1), self.get_n_players(), replace=False)
        for i in range(self.get_n_players()):
            agent_pos = (all_agents_pos[i]//self.n_squares_width, all_agents_pos[i]%self.n_squares_width)
            state[2, agent_pos[0], agent_pos[1]] = i+1
        self.state = state
        self.steps_beyond_done = None
        self.players_in_game = np.ones(self.get_n_players())
        self.players_moved = np.zeros(self.get_n_players())
        return [self.state_to_observation(np.copy(self.state), player.player_id) for player in self.players]

    def step(self, action, player_id):
        #print('Player id', player_id)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        assert self.players_moved[player_id-1]==0
        agent_pos = list(zip(*np.where(self.state[2] == player_id)))[0]

        #check if player in game
        if self.players_in_game[player_id-1]==0:
            logger.warn("This player is out of the game.")
            reward, done, game_over = 0.0, 1, 0
            obs = self.state_to_observation(np.copy(self.state), player_id)
            return obs, reward, done, game_over, {}
        
        #move
        new_agent_pos = np.array(agent_pos)
        if action==0:
            new_agent_pos[1]-=1
        elif action==1:
            new_agent_pos[1]+=1
        elif action==2:
            new_agent_pos[0]-=1
        elif action==3:
            new_agent_pos[0]+=1
        new_agent_pos = np.clip(new_agent_pos, 0, self.n_squares_width-1)

        #check if done - crashed into other snake or trail/wall
        done=False
        if self.state[2, new_agent_pos[0], new_agent_pos[1]]!=0 or self.state[1, new_agent_pos[0], new_agent_pos[1]]!=0:
            done=True
        
        if not done:
            reward=1
            self.state[2, agent_pos[0], agent_pos[1]] = 0 #moved from this position so it is empty
            self.state[2, new_agent_pos[0], new_agent_pos[1]] = player_id #moved to this position
            self.state[1, agent_pos[0], agent_pos[1]] = player_id #add wall at the location we just were
        else:
            reward=0
            self.players_in_game[player_id-1]=0    
        self.players_moved[player_id-1]=1

        #print('Moved', self.players_moved)
        #print('Ingame',self.players_in_game)

        turn_over = False
        if sum(self.players_moved)>=sum(self.players_in_game):
            turn_over=True
        if turn_over:
            self.players_moved = np.zeros(self.get_n_players())
        game_over=False
        if sum(self.players_in_game)==0:
               game_over=True
        if game_over:
            if self.steps_beyond_done is None: #everyone has just taken their turn
                self.steps_beyond_done = 0
            else:
                if self.steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1
            reward, done = 0.0, 1
        
        obs = self.state_to_observation(np.copy(self.state), player_id)
        return obs, reward, done, game_over, {}

    def state_to_observation(self, state, player_id):
        agent_pos = list(zip(*np.where(state[2] == player_id)))[0]
        agent_pos_channel = np.zeros((1, state.shape[1], state.shape[2]))
        agent_pos_channel[0, agent_pos[0], agent_pos[1]] = 1
        state[2, agent_pos[0], agent_pos[1]]=0
        state[state!=0] = 1
        observation = np.concatenate((state, agent_pos_channel), 0)
        return observation

    def render(self, values=None, mode='human'):
        screen_width = 800
        screen_height = 800

        square_size_height = screen_height/self.n_squares_height
        square_size_width = screen_width/self.n_squares_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #add invisible squares for visualising state values
            l, r, t, b = -square_size_width/2, square_size_width/2, square_size_height/2, -square_size_height/2
            self.squares = [[rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]) for j in range(0, self.n_squares_width)] for i in range(0, self.n_squares_height)]
            sq_transforms = [[rendering.Transform() for j in range(0, self.n_squares_width)] for i in range(0, self.n_squares_height)]
            for i in range(0, self.n_squares_height):
                for j in range(0, self.n_squares_width):
                    self.squares[i][j].add_attr(sq_transforms[i][j])
                    self.viewer.add_geom(self.squares[i][j])
                    sq_x, sq_y = self.convert_pos_to_xy((i, j), (square_size_width, square_size_height))
                    sq_transforms[i][j].set_translation(sq_x, sq_y)
                    self.squares[i][j].set_color(0, 0, 0)
            #horizontal grid lines
            '''for i in range(1, self.n_squares_height):
                track = rendering.Line((0,i*square_size_height), (screen_width,i*square_size_height))
                track.set_color(1,1,1)
                self.viewer.add_geom(track)
            #vertical grid lines
            for i in range(1, self.n_squares_width):
                track = rendering.Line((i*square_size_width, 0), (i*square_size_width, screen_height))
                track.set_color(1,1,1)
                self.viewer.add_geom(track)'''
            #add agents
            self.all_agents = []
            self.all_agents_trans = []
            for i in range(self.get_n_players()):
                agent = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                agent.set_color(*self.agent_cols[i])
                #agent.set_color(1, 1, 1)
                agenttrans = rendering.Transform()
                agent.add_attr(agenttrans)
                self.viewer.add_geom(agent)
                self.all_agents.append(agent)
                self.all_agents_trans.append(agenttrans)
            #the goal
            '''self.goal = make_oval(width=square_size_width/4, height=square_size_height/4)
            self.goal.set_color(1,0,1)
            self.goaltrans = rendering.Transform()
            self.goal.add_attr(self.goaltrans)
            self.viewer.add_geom(self.goal)'''
        if self.state is None: return
        '''goal_pos = list(zip(*np.where(self.state[0] == 1)))[0]'''
        for i, row in enumerate(self.state[1, :, :]):
            for j, val in enumerate(row):
                if val-1>=0:
                    self.squares[i][j].set_color(*self.agent_cols[val-1])
                else:
                    self.squares[i][j].set_color(0, 0, 0)
        for i in range(self.get_n_players()):
            agent_pos = list(zip(*np.where(self.state[2] == i+1)))[0]
            agent_x, agent_y = self.convert_pos_to_xy(agent_pos, (square_size_width, square_size_height))
            self.all_agents_trans[i].set_translation(agent_x, agent_y)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def convert_pos_to_xy(self, pos, size):
        x = (pos[1]+0.5) * size[0]
        y = (self.n_squares_height-pos[0]-0.5) * size[1]
        return x, y

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class Player():
    def __init__(self, env):
        self.env = env
        self.env.players.append(self)
        self.env.agent_cols.append(tuple(np.random.rand(3)))
        self.player_id = len(self.env.players)
        self.prev_action = None
    def pick_action(self, observation):
        raise NotImplementedError
        return action
    def step(self, action):
        #prevents crashing into itself
        if action==0 and self.prev_action==1:
            action = 1
        elif action==1 and self.prev_action==0:
            action = 0
        elif action==2 and self.prev_action==3:
            action = 3
        elif action==3 and self.prev_action==2:
            action = 2
        obs, reward, done, game_over, info = self.env.step(action, self.player_id)
        self.prev_action = action
        if done or game_over:
            self.prev_action=None
        return obs, reward, done, game_over, info
    def move(self, obs):
        action = self.pick_action(obs)
        self.step(action)
