# Hummingbird Parameter File
import numpy as np
# Initial Conditions
phi0 = 0.0 * np.pi / 180  # roll angle in rads
theta0 = 0 * np.pi / 180  # pitch angle in rads
psi0 = 0.0 * np.pi / 180  # yaw angle in rads
phidot0 = 0.0              # roll rate in rads/sec
thetadot0 = 0.0         # pitch rate in rads/sec
psidot0 = 0.0              # yaw rate in rads/sec
# Physical parameters of the hummingbird known to the controller
g = 9.81
ell1 = 0.247
ell2 = -0.039
ell3x = -0.007
ell3y = -0.007
ell3z = 0.018
ellT = 0.355
d = 0.12
m1 = 0.108862
J1x = 0.000189
J1y = 0.001953
J1z = 0.001894
m2 = 0.4717
J2x = 0.00231 # unclear what J2x is actually supposed to be
#J2x = 0.000231 # this is the value in the Hummingbird Manual, but it seems wrong
J2y = 0.003274
J2z = 0.003416
m3 = 0.1905
J3x = 0.0002222
J3y = 0.0001956
J3z = 0.000027

# Helpful parameters that only need calculated once
# Calculate equilibrium force
thetaE = 0.0  # equilibrium angle
Fe = (m1*ell1 + m2*ell2)*g*np.cos(thetaE)/ellT # Equation on pg 24
JT = m1 * ell1**2 + m2 * ell2**2 + J2z + m3 * (ell3x**2 + ell3y**2) # Equation on pg 25
b_theta = ellT/(m1 * ell1**2 + m2 * ell2**2 + J1y + J2y) # see eqn 4.5 in manual
b_psi = ellT * Fe / (JT + J1z) # Equation 5.4 in Hummingbird manual

# u_l_e and u_r_e scale is percent duty cycle [0, 1]
# A reasonable setting for u_l_e and u_r_e would be that, at equilibrium, we allow for
# half the throttle response in either direction. Hence, u_l_e = u_r_e = 0.5
u_l_e = 0.5
u_r_e = 0.5
# Correct version of km (see equation above 4.9 in hummingbird manual)
km = g * (m1 * ell1 + m2 * ell2) / (ellT * (u_l_e + u_r_e))
# The below is km if u_l_e + u_r_e = 1
#km = g * (m1 * ell1 + m2 * ell2) / ellT  # need to find this experimentally for hardware

# mixing matrix
unmixing = np.array([[1.0, 1.0], [d, -d]]) # converts fl and fr (LR) to force and torque (FT)
mixing = np.linalg.inv(unmixing) # converts force and torque (FT) to fl and fr (LR) 

# Simulation Parameters
t_start = 0.0  # Start time of simulation
t_end = 100.0  # End time of simulation
Ts = 0.01  # sample time for simulation
t_plot = 0.1  # the plotting and animation is updated at this rate
# saturation limits
force_max = 2.0                # Max force N
torque_max = 5.0                # Max torque, Nm

