import time
from snaky_env import *

env = SnakyEnv(80)

# multiplayer
'''players = [Player(env) for _ in range(55)]
env.reset()

game_over=False
step=0
while not game_over:
    print(step)
    for i in range(env.get_n_players()):
        if env.players_in_game[i]==1:
            obs, reward, done, game_over, info = env.players[i].step(np.random.randint(4))
    env.render()
    step+=1
    
env.render()
'''

# single player
p1 = Player(env)
env.reset()

game_over=False
step=0
while not game_over:
    obs, reward, done, game_over, info = p1.step(np.random.randint(4))
    env.render()
    step+=1
    
env.render()
