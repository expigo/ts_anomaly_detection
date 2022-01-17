import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from pso.particle import Particle


class PSO:
    def __init__(self, swarm_size=16, N=10):
        self.swarm_size = swarm_size
        self.n_iters = N

        self.particles = np.array([])

        self.best_position_history = []
        self.best_fitness_history = []

    def run_old(self, fitness_fn, space):

        self.create_swarm(fitness_fn, space)

        for gen in range(self.n_iters):
            print(f">>> generation: {gen} <<< ")
            Parallel(n_jobs=4, require='sharedmem')(delayed(evolve)(particle, self.best_position_history[-1]) for particle in self.particles)
            # for particle in self.particles:
            #     particle.update_velocity_canonical(gbest=self.best_position_history[-1])
            #     # particle.update_velocity(gbest=self.best_position_history[-1], c1=.5, c2=.5, omega=.5)
            #     particle.update_position()

            self.evaluate_swarm()

        return self.best_position_history[-1], self.best_fitness_history[-1]

    def run(self, fitness_fn, space):

        self.create_swarm(fitness_fn, space)

        for gen in range(self.n_iters):
            for i, p in enumerate(self.particles):
                print(f' ***** [{i}/{gen}] ***** ')
                # p.update_velocity_canonical(gbest=self.best_position_history[-1])
                p.update_velocity(gbest=self.best_position_history[-1], c1=.5, c2=.5, omega=.5)
                p.update_position()

                if p < self.best_fitness_history[-1]:
                    self.best_position_history.append(p.current_position)
                    self.best_fitness_history.append(p.fitness_history[-1])
                else:
                    self.best_position_history.append(self.best_position_history[-1])
                    self.best_fitness_history.append(self.best_fitness_history[-1])

        return self.best_position_history[-1], self.best_fitness_history[-1]

    def create_swarm(self, fitness_fn, space_def: dict):
        self.particles = np.array([Particle.from_space(space_def=space_def, fitness_fn=fitness_fn)
                                   for _ in range(self.swarm_size)])

        # self.particles = np.array(Parallel(n_jobs=4)(delayed(create_particle)(space_def, fitness_fn)
        #                                              for _ in range(self.swarm_size)))

        # self.evaluate_swarm()

        gen_best = self.particles.min()
        gen_best_fitness = gen_best.fitness_history[-1]
        gen_best_position = gen_best.current_position
        self.best_position_history.append(gen_best_position)
        self.best_fitness_history.append(gen_best_fitness)

    def evaluate_swarm(self):
        gen_best = self.particles.min()
        gen_best_fitness = gen_best.fitness_history[-1]
        gen_best_position = gen_best.current_position

        if gen_best_fitness < self.best_fitness_history[-1]:
            self.best_position_history.append(gen_best_position)
            self.best_fitness_history.append(gen_best_fitness)
        else:
            self.best_position_history.append(self.best_position_history[-1])
            self.best_fitness_history.append(self.best_fitness_history[-1])

    def plot_fitness_history(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.best_fitness_history)
        # plt.plot(self.best_fitness_history)
        # plt.show()


def evolve(particle: Particle, gbest):
    # print('>>>', particle.current_position.raw)
    # particle.update_velocity_canonical(gbest=gbest)
    particle.update_velocity(gbest=gbest, c1=.5, c2=.5, omega=.5)
    particle.update_position()


def create_particle(space_def, fitness_fn):
    return Particle.from_space(space_def=space_def, fitness_fn=fitness_fn)
