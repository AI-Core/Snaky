import time, random, string
from snaky_env import *

env = SnakyEnv(50)

# multiplayer

players = [Player(env, "".join(random.choices(string.ascii_lowercase, k=6))) for _ in range(5)]
for _ in range(10):
    env.reset()
    game_over=False
    step=0
    while not game_over:
        print(step)
        for i in range(env.get_n_players()):
            if not game_over:
                if env.players_in_game[i]==1:
                    obs, reward, done, game_over, info = env.players[i].step(np.random.randint(4))
        env.render()
        step+=1 
    env.render()


# single player
'''
p1 = Player(env)
env.reset()
game_over=False
step=0
while not game_over:
    obs, reward, done, game_over, info = p1.step(np.random.randint(4))
    env.render()
    step+=1
env.render()
'''
