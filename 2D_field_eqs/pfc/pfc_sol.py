import numpy as np
import dedalus.public as d3
from dedalus.core import operators
from dedalus.extras.plot_tools import plot_bot_2d
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# Parameters
q0 = 1.0
epsilon = 1e-1
Lx, Ly = 1.0, 1.0
Nx, Ny = 128, 128
dtype = np.float64
dealias = 3 / 2
stop_sim_time = 0.02
timestepper = d3.RK222
timestep = 1e-5


def power(A, B):
    # Operators
    return operators.Power(A, B)


# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)

# Vector Field
u = dist.Field(name='u', bases=(xbasis, ybasis))

# Problem
problem = d3.IVP([u], namespace=locals())
# TODO: Please Check this
# Eq: dt(u) = lap(U^2) + lap(U^3) + lap(q_0*U + 2q_0*lap(U) +lap(lap(U)) - epsilon U)
problem.add_equation(
    "dt(u) -q0*lap(u)= lap(power(u,2)) + lap(power(u,3)) + 2*q0*lap(lap(u)) + lap(lap(lap(u))) - epsilon*lap(u)")

# Initial conditions
np.random.seed(0)
u['g'] = np.random.randn(*u['g'].shape)

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time


# Main loop
u_list = []
while solver.proceed:
    solver.step(timestep)
    if solver.iteration % 10 == 0:
        u.change_scales(1)
        u_list.append(np.copy(u['g']))
    if solver.iteration % 1000 == 0:
        print('Completed iteration {}'.format(solver.iteration))

# Convert storage lists to arrays
u_array = np.array(u_list)
np.save(f'u_cahn_hilliard_Nx_{Nx}_Ny_{Ny}_timestep_{timestep}', u_array)
