import numpy as np
import hummingbirdParam as P


class ctrlStateFeedbackIntegrator:
    def __init__(self):
        # DC Gains
        self.k_th    = 4.0842513
        self.k_thdot = 0.59561998
        self.ki_lon  = -9.076113997881688

        # Lateral gains
        self.k_phi     = 0.0489888
        self.k_psi     = 0.10039459
        self.k_phidot  = 0.0043848
        self.k_psidot  = 0.03965126
        self.ki_lat    = -0.09210513161845148


        # dirty dreivative setup
        self.Ts = P.Ts
        sigma = 0.05
        self.beta = (2.0 * sigma - self.Ts) / (2.0 * sigma + self.Ts)

        # Previous samples
        self.phi_d1   = 0.0
        self.theta_d1 = 0.0
        self.psi_d1   = 0.0

        # Estimated derivatives
        self.phi_dot   = 0.0
        self.theta_dot = 0.0
        self.psi_dot   = 0.0

        # Integrators and error terms
        self.integrator_th  = 0.0
        self.error_th_d1    = 0.0

        self.integrator_psi = 0.0
        self.error_psi_d1   = 0.0

        # constant distrubances
        self.force_disturbance  = 0.0
        self.torque_disturbance = 0.0

    def update(self, r: np.ndarray, y: np.ndarray):
        # unpack reference and measurements
        theta_ref = r[0, 0]
        psi_ref   = r[1, 0]

        phi   = y[0, 0]
        theta = y[1, 0]
        psi   = y[2, 0]

        force_equilibrium = P.Fe

        # Dirty derivatives
        self.phi_dot = self.beta * self.phi_dot \
            + (1.0 - self.beta) * (phi - self.phi_d1) / self.Ts
        self.phi_d1 = phi

        # theta_dot
        self.theta_dot = self.beta * self.theta_dot \
            + (1.0 - self.beta) * (theta - self.theta_d1) / self.Ts
        self.theta_d1 = theta

        # psi_dot
        self.psi_dot = self.beta * self.psi_dot \
            + (1.0 - self.beta) * (psi - self.psi_d1) / self.Ts
        self.psi_d1 = psi

        # integrators
        error_th  = theta_ref - theta
        error_psi = psi_ref   - psi

        self.integrator_th  += (self.Ts / 2.0) * (error_th  + self.error_th_d1)
        self.integrator_psi += (self.Ts / 2.0) * (error_psi + self.error_psi_d1)

        self.error_th_d1  = error_th
        self.error_psi_d1 = error_psi

        # Longitudinal control (theta)
        F_tilde = -(
            self.k_th    * theta +
            self.k_thdot * self.theta_dot +
            self.ki_lon  * self.integrator_th
        )

        force_unsat = force_equilibrium + F_tilde + self.force_disturbance
        force = saturate(force_unsat, 0.0, P.force_max)

        # Lateral control (phi, psi)
        tau_unsat = -(
            self.k_phi    * phi +
            self.k_psi    * psi +
            self.k_phidot * self.phi_dot +
            self.k_psidot * self.psi_dot +
            self.ki_lat   * self.integrator_psi
        ) + self.torque_disturbance

        torque = saturate(tau_unsat, -P.torque_max, P.torque_max)

        # Convert force and torque to PWM signals
        pwm = np.array([
            [force + torque / P.d],
            [force - torque / P.d]
        ]) / (2.0 * P.km)

        pwm = saturate(pwm, 0.0, 1.0)

        # Reference vector for plotting
        y_ref = np.array([[0.0], [theta_ref], [psi_ref]])

        return pwm, y_ref


def saturate(u, low_limit, up_limit):
    if isinstance(u, float) or isinstance(u, np.floating):
        if u > up_limit:
            u = up_limit
        if u < low_limit:
            u = low_limit
    else:
        for i in range(u.shape[0]):
            if u[i, 0] > up_limit:
                u[i, 0] = up_limit
            if u[i, 0] < low_limit:
                u[i, 0] = low_limit
    return u
