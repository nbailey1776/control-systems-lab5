import numpy as np
import hummingbirdParam as P

def saturate(u, low, up):
    if isinstance(u, (float, np.floating)):
        if u > up:
            return up
        if u < low:
            return low
        return u
    u = np.array(u, copy=True)
    for i in range(u.shape[0]):
        if u[i, 0] > up:
            u[i, 0] = up
        if u[i, 0] < low:
            u[i, 0] = low
    return u

class ctrlPID:
    def __init__(self):
        # timing / filter
        self.Ts = P.Ts
        self.sigma = 0.05
        self.beta = (2*self.sigma - self.Ts) / (2*self.sigma + self.Ts)

        # Longitudinal theta PID
        tr_theta = 1.0
        zeta_theta = 0.707
        b_theta = P.ellT / (P.m1*P.ell1**2 + P.m2*P.ell2**2 + P.J1y + P.J2y)
        wn_theta = 2.2 / tr_theta

        self.kp_theta = (wn_theta**2) / b_theta
        self.kd_theta = (2*zeta_theta*wn_theta) / b_theta
        # small integral gain
        self.ki_theta = 0.5 * self.kp_theta * 0.01

        # equilibrium force (linearized)
        self.Fe = (P.m1*P.ell1 + P.m2*P.ell2) * P.g / P.ellT

        # Inner roll (phi) PD base
        tr_phi = 0.1
        zeta_phi = 0.707
        wn_phi = 2.2 / tr_phi
        self.kp_phi = (wn_phi**2) * P.J1x
        self.kd_phi = (2*zeta_phi*wn_phi) * P.J1x

        # Outer yaw (psi) PID gains
        tr_psi = 1.0
        zeta_psi = 0.707
        wn_psi = 2.2 / tr_psi

        JT = (P.m1*P.ell1**2
              + P.m2*P.ell2**2
              + P.J2z
              + P.m3*(P.ell3x**2 + P.ell3y**2))
        Fe_lin = self.Fe
        b_psi = (P.ellT * Fe_lin) / (JT + P.J1z)

        self.kp_psi = (wn_psi**2) / b_psi
        self.kd_psi = (2*zeta_psi*wn_psi) / b_psi
        # small integral gain for yaw
        self.ki_psi = 0.5 * self.kp_psi * 0.01

        # memory for derivatives
        self.theta_dot = 0.0
        self.phi_dot = 0.0
        self.psi_dot = 0.0
        self.theta_d1 = 0.0
        self.phi_d1 = 0.0
        self.psi_d1 = 0.0
        self.theta_int = 0.0
        self.psi_int = 0.0
        self.e_theta_d1 = 0.0
        self.e_psi_d1 = 0.0

    def update(self, r, y):
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
        phi_des = (self.kp_psi*e_psi
                   - self.kd_psi*self.psi_dot
                   + self.ki_psi*self.psi_int)
        psi_int_candidate = self.psi_int + 0.5*self.Ts*(e_psi + self.e_psi_d1)

        # Inner Roll Loop: creates torque tau
        e_phi = phi_des - phi
        tau_unsat = self.kp_phi*e_phi - self.kd_phi*self.phi_dot
        tau = saturate(tau_unsat, -P.torque_max, P.torque_max)

        # yaw anti-windup
        if abs(tau - tau_unsat) < 1e-6:
            self.psi_int = psi_int_candidate

        # longitudinal pitch loop: creates force F_cmd
        e_theta = theta_ref - theta

        F_tilde = (self.kp_theta*e_theta
                   - self.kd_theta*self.theta_dot
                   + self.ki_theta*self.theta_int)

        F_unsat = self.Fe + F_tilde
        F_cmd = saturate(F_unsat, -P.force_max, P.force_max)

        theta_int_candidate = self.theta_int + 0.5*self.Ts*(e_theta + self.e_theta_d1)

        # pitch anti-windup
        if abs(F_cmd - F_unsat) < 1e-6:
            self.theta_int = theta_int_candidate

        # Mixing to PWM commands
        pwm = np.array([
            [F_cmd + tau / P.d],
            [F_cmd - tau / P.d]
        ]) / (2.0 * P.km)
        pwm = saturate(pwm, 0.0, 1.0)

        # save histories
        self.theta_d1 = theta
        self.phi_d1 = phi
        self.psi_d1 = psi
        self.e_theta_d1 = e_theta
        self.e_psi_d1 = e_psi

        y_ref = np.array([[phi_des],
                          [theta_ref],
                          [psi_ref]])

        return pwm, y_ref
