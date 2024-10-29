import numpy as np
import pygame

from src.helpers.constants import COLORS, SCREEN
from src.helpers.data_structures import Point


def set_boundaries():
    boundaries: list[Point] = []

    for i in range(1, SCREEN['WIDTH'] - 1):
        boundaries.extend([Point(i, 0), Point(i, SCREEN['HEIGHT'] - 1)])
    for i in range(1, SCREEN['HEIGHT'] - 1):
        boundaries.extend([Point(0, i), Point(SCREEN['WIDTH'] - 1, i)])

    return boundaries


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (SCREEN['WIDTH'] * SCREEN['GRID_SIZE'], SCREEN['HEIGHT'] * SCREEN['GRID_SIZE']))
        pygame.display.set_caption("Snake")

        self.boundaries = set_boundaries()
        self.player = Player()
        self.input_direction = np.array([0, 1, 0])
        self.food: Point
        self.score = 0
        self.frame_iteration = 0
        self._place_food()

    def play_step(self, agent_input):
        reward = 0
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self.input_direction = np.array(agent_input, int)

        self.player.move(turn_direction=self.input_direction)
        self.input_direction = np.array([0, 1, 0])  # reset direction

        if self.is_death_collision() or (self.frame_iteration > len(self.player.get_player()) * 100):
            # self.reset()
            reward = -10
            return reward, True, self.score

        if self._is_food_collision():
            self.player.grow()
            self._place_food()
            reward = 10
            self.score += 1

        # drawing
        self.screen.fill(COLORS['BLACK'])
        self._draw_boundaries()
        self._draw_player()
        self._draw_food()
        pygame.display.flip()

        return reward, False, self.score

    def _draw_player(self):
        for p in self.player.get_player():
            self._draw_cell(p, color=COLORS['RED'])

    def _draw_cell(self, point, color=COLORS['GREEN']):
        pygame.draw.rect(self.screen, color, (point.x * SCREEN['GRID_SIZE'], point.y * SCREEN['GRID_SIZE'],
                                              SCREEN['GRID_SIZE'],
                                              SCREEN['GRID_SIZE']))

    def _draw_boundaries(self):
        for b in self.boundaries:
            self._draw_cell(b, color=COLORS['BLUE'])

        # draw corners of a screen
        self._draw_cell(Point(0, 0), color=COLORS['BLUE'])
        self._draw_cell(Point(SCREEN['WIDTH'] - 1, 0), color=COLORS['BLUE'])
        self._draw_cell(Point(SCREEN['WIDTH'] - 1, SCREEN['HEIGHT'] - 1), color=COLORS['BLUE'])
        self._draw_cell(Point(0, SCREEN['HEIGHT'] - 1), color=COLORS['BLUE'])

    def _draw_food(self):
        self._draw_cell(self.food, color=COLORS['YELLOW'])

    def is_death_collision(self, future_point=None):
        if future_point is not None:
            if future_point in self.boundaries or future_point in self.player.tail:
                return True
            return False

        if self.player.head in self.boundaries or self.player.head in self.player.tail:
            return True
        return False

    def _is_food_collision(self):
        if self.player.head == self.food:
            return True
        return

    def _place_food(self):
        self.food = Point(np.random.randint(1, SCREEN['WIDTH'] - 2), np.random.randint(1, SCREEN['HEIGHT'] - 2))

    def reset(self):
        self.player = Player()
        self.input_direction = np.array([0, 1, 0])
        self.food: Point
        self.score = 0
        self.frame_iteration = 0
        self._place_food()


class Player:
    def __init__(self):
        self.head = Point(SCREEN['WIDTH'] // 2, SCREEN['HEIGHT'] // 2)
        self.tail = [self.head - Point(1, 0), self.head - Point(2, 0)]
        self.body = [self.head] + self.tail

        self.current_direction = np.array([0, 0, 1, 0], int)  # [west, north, east, south] of a screen

    def move(self, turn_direction=np.array([0, 1, 0])):
        """
        :param turn_direction: [left, straight, right]
        :return:
        """

        # set new direction
        # if np.array_equal(turn_direction, np.array([0, 1, 0])):
        #     pass
        if np.array_equal(turn_direction, np.array([1, 0, 0])):
            self.current_direction = np.roll(self.current_direction, -1)
        elif np.array_equal(turn_direction, np.array([0, 0, 1])):
            self.current_direction = np.roll(self.current_direction, 1)

        # move player
        if self.current_direction[0] == 1:  # west
            self.head += Point(-1, 0)
        elif self.current_direction[1] == 1:  # north
            self.head += Point(0, -1)
        elif self.current_direction[2] == 1:  # east
            self.head += Point(1, 0)
        elif self.current_direction[3] == 1:  # south
            self.head += Point(0, 1)

        self.tail = self.body[:-1]
        self.body = [self.head] + self.tail

    def grow(self):
        added_point = self.body[-1]
        self.body.append(added_point)

    def get_player(self):
        return self.body
