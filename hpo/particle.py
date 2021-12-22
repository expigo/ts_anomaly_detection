import numpy as np

from hpo.vector import Vector

rng = np.random.default_rng()


class Particle:
    def __init__(self, initial_position: Vector, fitness_fn):
        self.current_position = initial_position
        self.fitness_fn = fitness_fn
        self.position_history = []
        self.position_best_history = []
        self.fitness_history = []
        self.fitness_best_history = []
        self.velocity_history = []

        # TODO: assert lower bound <= upper bound

        dimensions = len(self.current_position)

        initial_velocity = initial_position.draw_random_like()
        self.velocity_history.append(initial_velocity.raw)

        self.position_history.append(initial_position)
        self.position_best_history.append(initial_position)

        initial_fitness = self.evaluate()
        self.fitness_history.append(initial_fitness)
        self.fitness_best_history.append(initial_fitness)


    def evaluate(self, position=None, fitness_fn=None):
        if position is None:
            position = self.position_history[-1]
        if fitness_fn is None:
            fitness_fn = self.fitness_fn

        fitness = fitness_fn(position.raw)
        return fitness

    def update_velocity_canonical(self, gbest, c1=2.05, c2=2.05):
        fi = c1 + c2
        K = 2 / np.abs(2 - fi - np.sqrt(np.power(fi, 2) - 4 * fi))

        dims = self.current_position.shape
        r1 = rng.uniform(size=dims)
        r2 = rng.uniform(size=dims)

        cognitive_velocity = c1 * r1 * (self.position_best_history[-1].raw - self.current_position.raw)
        social_velocity = c2 * r2 * (gbest.raw - self.current_position.raw)

        new_velocity = K * (self.velocity_history[-1] + cognitive_velocity + social_velocity)
        self.velocity_history.append(new_velocity)

    def update_velocity(self, gbest, c1=2.05, c2=2.05, omega=.5):
        dims = self.current_position.shape
        r1 = rng.uniform(size=dims)
        r2 = rng.uniform(size=dims)

        cognitive_velocity = c1 * r1 * (self.position_best_history[-1].raw - self.current_position.raw)
        social_velocity = c2 * r2 * (gbest.raw - self.current_position.raw)

        new_velocity = omega * self.velocity_history[-1] + cognitive_velocity + social_velocity
        self.velocity_history.append(new_velocity)

    def update_position(self):

        current_best_position = self.position_best_history[-1]
        current_best_fitness = self.fitness_best_history[-1]

        new_position = self.current_position + self.velocity_history[-1]

        if self.current_position == new_position:
            new_fitness = current_best_fitness
        else:
            self.current_position = new_position
            new_fitness = self.evaluate(position=new_position)

        self.position_history.append(new_position)
        self.fitness_history.append(new_fitness)

        if new_fitness < current_best_fitness:
            current_best_fitness = new_fitness
            current_best_position = new_position

        self.position_best_history.append(current_best_position)
        self.fitness_best_history.append(current_best_fitness)

    def __lt__(self, other):
        if isinstance(other, Particle):
            return self.fitness_history[-1] < other.fitness_history[-1]
        else:
            return self.fitness_history[-1] < other

    def __le__(self, other):
        if isinstance(other, Particle):
            return self.fitness_history[-1] <= other.fitness_history[-1]
        else:
            return self.fitness_history[-1] <= other

    @classmethod
    def from_space(cls, space_def, fitness_fn):
        position = Vector.from_definition(space_def)
        return cls(initial_position=position, fitness_fn=fitness_fn)
