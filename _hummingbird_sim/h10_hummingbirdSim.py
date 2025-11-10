import matplotlib.pyplot as plt
import numpy as np
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdAnimation import HummingbirdAnimation
from dataPlotter import DataPlotter
from hummingbirdDynamics import HummingbirdDynamics
from ctrlPID import ctrlPID, saturate

# instantiate pendulum, controller, and reference classes
hummingbird = HummingbirdDynamics(alpha=0.2)
controller = ctrlPID()
psi_ref = SignalGenerator(amplitude=30.*np.pi/180., frequency=0.02)
theta_ref = SignalGenerator(amplitude=15.*np.pi/180., frequency=0.05)

# instantiate the simulation plots and animation
dataPlot = DataPlotter()
animation = HummingbirdAnimation()

t = P.t_start  # time starts at t_start
y = hummingbird.h()
F_step_time = 10.0 # time to apply force disturbance
F_disturbance = 1.0 # 1 N step in net force
while t < P.t_end:  # main simulation loop

    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        r = np.array([[theta_ref.square(t)],
                      [psi_ref.square(t)]])

        pwm_cmd, y_ref = controller.update(r, y)

        if t >= F_step_time:
            uL = pwm_cmd[0, 0]
            uR = pwm_cmd[1, 0]

            # net force from both motors
            F_nom = P.km * (uL + uR)
            # torque component from difference
            T_nom = P.km * P.d * (uL - uR)

            # add 1 N disturbance to net force
            F_dist = F_nom + F_disturbance

            sum_u = F_dist / P.km
            diff_u = T_nom / (P.km * P.d)

            uL_new = 0.5 * (sum_u + diff_u)
            uR_new = 0.5 * (sum_u - diff_u)

            pwm = np.array([[uL_new], [uR_new]])
            pwm = saturate(pwm, 0.0, 1.0)
        else:
            pwm = pwm_cmd

        y = hummingbird.update(pwm)
        t += P.Ts
    # update animation and data plots at rate t_plot
    animation.update(t, hummingbird.state)
    dataPlot.update(t, hummingbird.state, pwm, y_ref)

    # the pause causes figure to be displayed during simulation
    plt.pause(0.1)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
