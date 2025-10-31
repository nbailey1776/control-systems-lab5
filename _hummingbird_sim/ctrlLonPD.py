import numpy as np
import hummingbirdParam as P

class ctrlLonPD:
    def __init__(self):
        tr_pitch = 1.0 # desired rise time
        zeta_pitch = 0.707 # desired damping ratio

        # Plant constant b_theta from linearized model)
        b_theta = P.ellT / (P.m1 * P.ell1**2
                            + P.m2 * P.ell2**2
                            + P.J1y + P.J2y)

        # Natural frequency from rise time
        wn_pitch = 2.2 / tr_pitch

        # PD gains from pole placement
        self.kp_pitch = (wn_pitch**2) / b_theta
        self.kd_pitch = (2 * zeta_pitch * wn_pitch) / b_theta
        print('kp_pitch: ', self.kp_pitch)
        print('kd_pitch: ', self.kd_pitch)

        self.Ts = P.Ts
        # dirty derivative
        self.sigma = 0.05
        self.beta = (2 * self.sigma - self.Ts) / (2 * self.sigma + self.Ts)

        self.theta_d1 = 0.0 # last theta
        self.theta_dot = 0.0 # filtered derivative of theta
        self.error_theta_d1 = 0.0 # last pitch error

    def update(self, r: np.ndarray, y: np.ndarray):

        theta_ref = r[0][0] # desired pitch
        theta = y[1][0] # measured pitch

        # Pitch tracking error
        error_theta = theta_ref - theta

        # Dirty derivative for theta_dot
        self.theta_dot = self.beta * self.theta_dot + (1 - self.beta) * (theta - self.theta_d1) / self.Ts

        # Feedforward force to cancel gravity at current theta
        Ffl = ((P.m1 * P.ell1 + P.m2 * P.ell2) * P.g / P.ellT) * np.cos(theta)

        # PD control for force F_tilde
        F_tilde = (self.kp_pitch * error_theta
                   - self.kd_pitch * self.theta_dot)

        # Total commanded force
        force_unsat = Ffl + F_tilde

        # Saturate total force to what the motors can actually do
        force = saturate(force_unsat, -P.force_max, P.force_max)

        # No lateral torque
        torque = 0.0

        # Convert (force, torque) -> left/right PWM.

        pwm = np.array([
            [force + torque / P.d],
            [force - torque / P.d]
        ]) / (2 * P.km)

        pwm = saturate(pwm, 0.0, 1.0)

        # Save delayed values for next step
        self.theta_d1 = theta
        self.error_theta_d1 = error_theta

        # Return motor PWMs and reference vector for plotting
        y_ref = np.array([[0.0], [theta_ref], [0.0]])
        return pwm, y_ref


def saturate(u, low_limit, up_limit):
    if isinstance(u, float) or isinstance(u, np.floating):
        if u > up_limit:
            u = up_limit
        if u < low_limit:
            u = low_limit
    else:
        for i in range(0, u.shape[0]):
            if u[i][0] > up_limit:
                u[i][0] = up_limit
            if u[i][0] < low_limit:
                u[i][0] = low_limit
    return u
