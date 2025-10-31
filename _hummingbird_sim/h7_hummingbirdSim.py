import matplotlib.pyplot as plt
import numpy as np
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdAnimation import HummingbirdAnimation
from dataPlotter import DataPlotter
from hummingbirdDynamics import HummingbirdDynamics
from ctrlLonPD import ctrlLonPD

# instantiate pendulum, controller, and reference classes
hummingbird = HummingbirdDynamics(alpha=0.1)
controller = ctrlLonPD()
theta_ref = SignalGenerator(amplitude=0.5, frequency=0.1)

# instantiate the simulation plots and animation
dataPlot = DataPlotter()
animation = HummingbirdAnimation()

t = P.t_start  # time starts at t_start
y = hummingbird.h()
u = np.zeros((2, 1))
y_ref = np.zeros((3, 1))
while t < P.t_end:  # main simulation loop

    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        r = np.array([[theta_ref.square(t)], [0.0]])
        u, y_ref = controller.update(r, y)
        y = hummingbird.update(u)  # Propagate the dynamics
        t += P.Ts  # advance time by Ts

    # update animation and data plots at rate t_plot
    animation.update(t, hummingbird.state)
    dataPlot.update(t, hummingbird.state, u, y_ref)

    # the pause causes figure to be displayed during simulation
    plt.pause(0.001)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
