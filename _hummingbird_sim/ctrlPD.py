import numpy as np
import hummingbirdParam as P

class ctrlPD:
    def __init__(self):
        # Longitudinal PD gains
        tr_pitch = 1.0
        zeta_pitch = 0.707
        b_theta = P.ellT / (P.m1 * P.ell1**2
                            + P.m2 * P.ell2**2
                            + P.J1y + P.J2y)
        wn_pitch = 2.2 / tr_pitch
        self.kp_theta = (wn_pitch**2) / b_theta
        self.kd_theta = (2 * zeta_pitch * wn_pitch) / b_theta
        self.ki_theta = 0.0

        # Roll inner loop PD gains
        tr_phi = 0.1
        zeta_phi = 0.707
        wn_phi = 2.2 / tr_phi
        self.kp_phi = (wn_phi**2) * P.J1x
        self.kd_phi = (2 * zeta_phi * wn_phi) * P.J1x

        # Yaw outer loop PD gains
        tr_psi = 1.0
        zeta_psi = 0.707
        wn_psi = 2.2 / tr_psi

        JT = (P.m1 * P.ell1**2 + P.m2 * P.ell2**2
              + P.J2z + P.m3 * (P.ell3x**2 + P.ell3y**2))
        Fe = ((P.m1 * P.ell1 + P.m2 * P.ell2) * P.g) / P.ellT
        b_psi = (P.ellT * Fe) / (JT + P.J1z)
        self.kp_psi = (wn_psi**2) / b_psi
        self.kd_psi = (2 * zeta_psi * wn_psi) / b_psi

        # timing / dirty derivatives
        self.Ts = P.Ts
        self.sigma = 0.05
        self.beta = (2 * self.sigma - self.Ts) / (2 * self.sigma + self.Ts)

        # memory for derivatives
        self.theta_d1 = 0.0
        self.theta_dot = 0.0
        self.phi_d1 = 0.0
        self.phi_dot = 0.0
        self.psi_d1 = 0.0
        self.psi_dot = 0.0

    def update(self, r: np.ndarray, y: np.ndarray):

        phi   = y[0][0]
        theta = y[1][0]
        psi   = y[2][0]

        theta_ref = r[0][0]
        psi_ref   = r[1][0]

        # dirty derivatives for phi, theta, psi
        self.theta_dot = self.beta*self.theta_dot \
            + (1 - self.beta)*(theta - self.theta_d1)/self.Ts
        self.phi_dot = self.beta*self.phi_dot \
            + (1 - self.beta)*(phi - self.phi_d1)/self.Ts
        self.psi_dot = self.beta*self.psi_dot \
            + (1 - self.beta)*(psi - self.psi_d1)/self.Ts

        # Outer Yaw loop: creates desired roll angle phi_des
        e_psi = psi_ref - psi
        phi_des = (self.kp_psi * e_psi
                   - self.kd_psi * self.psi_dot)

        # Inner Roll Loop: creates torque tau
        e_phi = phi_des - phi
        tau_unsat = (self.kp_phi * e_phi
                     - self.kd_phi * self.phi_dot)

        tau = saturate(tau_unsat, -P.torque_max, P.torque_max)

        # Longitudinal Loop
        e_theta = theta_ref - theta

        # feedforward to hold up at current theta
        Ffl = ((P.m1 * P.ell1 + P.m2 * P.ell2) * P.g / P.ellT) * np.cos(theta)

        F_tilde = (self.kp_theta * e_theta
                   - self.kd_theta * self.theta_dot)

        F_unsat = Ffl + F_tilde
        F_cmd = saturate(F_unsat, -P.force_max, P.force_max)

        # Mix force+torque into left/right motor PWM
        pwm = np.array([
            [F_cmd + tau / P.d],
            [F_cmd - tau / P.d]
        ]) / (2 * P.km)

        pwm = saturate(pwm, 0.0, 1.0)

        # update delay states
        self.theta_d1 = theta
        self.phi_d1 = phi
        self.psi_d1 = psi

        y_ref = np.array([[phi_des],
                          [theta_ref],
                          [psi_ref]])
        return pwm, y_ref

def saturate(u, low_limit, up_limit):
    if isinstance(u, float) or isinstance(u, np.floating):
        if u > up_limit:
            u = up_limit
        if u < low_limit:
            u = low_limit
    else:
        for i in range(u.shape[0]):
            if u[i][0] > up_limit:
                u[i][0] = up_limit
            if u[i][0] < low_limit:
                u[i][0] = low_limit
    return u
