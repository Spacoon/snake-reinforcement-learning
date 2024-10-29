import random
from collections import deque

import numpy as np
import torch

from src.ai.model import Linear_QNet, QTrainer
from src.game.game import Game
from src.helpers.constants import TRAINING
from src.helpers.data_structures import Point
from src.helpers.plotter import plot


def get_state(game: Game):
    point_west = game.player.head + Point(-1, 0)
    point_north = game.player.head + Point(0, -1)
    point_east = game.player.head + Point(1, 0)
    point_south = game.player.head + Point(0, 1)

    dir_west = game.player.current_direction[0]
    dir_north = game.player.current_direction[1]
    dir_east = game.player.current_direction[2]
    dir_south = game.player.current_direction[3]

    state = [
        # danger straight
        (dir_west and game.is_death_collision(point_west)) or
        (dir_north and game.is_death_collision(point_north)) or
        (dir_east and game.is_death_collision(point_east)) or
        (dir_south and game.is_death_collision(point_south)),

        # danger right
        (dir_west and game.is_death_collision(point_north)) or
        (dir_north and game.is_death_collision(point_east)) or
        (dir_east and game.is_death_collision(point_south)) or
        (dir_south and game.is_death_collision(point_west)),

        # danger left
        (dir_west and game.is_death_collision(point_south)) or
        (dir_north and game.is_death_collision(point_west)) or
        (dir_east and game.is_death_collision(point_north)) or
        (dir_south and game.is_death_collision(point_east)),

        # move direction
        dir_west,
        dir_north,
        dir_east,
        dir_south,

        # direction to food
        game.player.head.x < game.food.x,
        game.player.head.x > game.food.x,
        game.player.head.y > game.food.y,
        game.player.head.y < game.food.y
    ]

    return np.array(state, dtype=int)


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=TRAINING['MAX_MEMORY'])
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=TRAINING['LR'], gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > TRAINING['BATCH_SIZE']:
            mini_sample = random.sample(self.memory, TRAINING['BATCH_SIZE'])
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.model.device)
            prediction = self.model(state0)  # calls the forward method
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game()
    while True:
        state_old = get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games} Score: {score} Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
