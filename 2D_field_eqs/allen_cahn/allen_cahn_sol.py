import numpy as np
import dedalus.public as d3
from dedalus.core import operators
from dedalus.extras.plot_tools import plot_bot_2d
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# Parameters
a1 = 1.0
a2 = 1.0
a3 = 1.0
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
# Eq: dt(u) = a1*lap(u) + a2*u - a3*u^3
problem.add_equation("dt(u) - a1*lap(u) - a2*u = -a3*power(u,3)")

# Initial conditions
# u['g'] = np.exp((1 - y**2) * np.cos(x + np.cos(x) * y**2)) * \
# (1 + 0.05 * np.cos(10 * (x + 2 * y)))
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
np.save('u', u_array)
