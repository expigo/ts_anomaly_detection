import numpy as np

from hpo.utils import plot_pso, Sphere, AlpineN2
from hpo.pso import PSO


N = 100
SS = 4
pso = PSO(swarm_size=SS, N=N)
function = AlpineN2(2)
position, fitness = pso.run_old(fitness_fn=lambda X: function(X),
                                space=[{
                                "low": function.input_domain[0][0],
                                "high": function.input_domain[0][1],
                                "type": "continuous",
                                "repeat": 2
                            },
                                # {
                                #     "low": 0,
                                #     "high": 10,
                                #     "type": "discrete",
                                #     "repeat": 1
                                # }
                            ])


plot_pso(function=function, pso=pso)
