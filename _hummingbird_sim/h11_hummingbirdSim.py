import matplotlib.pyplot as plt
import numpy as np
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdAnimation import HummingbirdAnimation
from dataPlotter import DataPlotter
from hummingbirdDynamics import HummingbirdDynamics
from ctrlStateFeedbackIntegrator import ctrlStateFeedbackIntegrator


def main():
    # Instantiate dynamics and controller
    hummingbird = HummingbirdDynamics(alpha=0.1)
    controller = ctrlStateFeedbackIntegrator()

    # Reference generators
    psi_ref = SignalGenerator(amplitude=30.0 * np.pi / 180.0, frequency=0.02)
    theta_ref = SignalGenerator(amplitude=15.0 * np.pi / 180.0, frequency=0.05)

    # Visualization
    dataPlot = DataPlotter()
    animation = HummingbirdAnimation()

    t = P.t_start
    y = hummingbird.h()

    while t < P.t_end:
        t_next_plot = t + P.t_plot
        while t < t_next_plot:
            r = np.array([
                [theta_ref.square(t)],
                [psi_ref.square(t)]
            ])

            pwm, y_ref = controller.update(r, y)
            y = hummingbird.update(pwm)
            t += P.Ts

        animation.update(t, hummingbird.state)
        dataPlot.update(t, hummingbird.state, pwm, y_ref)
        plt.pause(0.01)

    print("Press key to close")
    plt.waitforbuttonpress()
    plt.close()


if __name__ == "__main__":
    main()
