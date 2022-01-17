import numpy as np

from utils import plot_pso, Sphere, AlpineN2
from pso import PSO


N = 100
SS = 32
pso = PSO(swarm_size=SS, N=N)
function = AlpineN2(2)
position, fitness = pso.run(fitness_fn=lambda X: function(X),
                                space=[{
                                "low": function.input_domain[0][0],
                                "high": function.input_domain[0][1],
                                "space": "continuous",
                                "repeat": 1
                            },{
                                    "value": 5,
                                    "space": "constant",
                                    "repeat": 1
                                },
                                # {
                                #     "low": 0,
                                #     "high": 10,
                                #     "type": "discrete",
                                #     "repeat": 1
                                # }
                            ])


plot_pso(function=function, pso=pso)
