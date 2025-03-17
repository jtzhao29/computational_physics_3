import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

# Define the system of ODEs
def equations(t, u, p):
    theta, Omega = u
    a, l, m, g, omega = p
    dtheta_dt = Omega
    dOmega_dt = (a / l) * omega**2 * np.cos(omega * t) * np.sin(theta) - (g / l) * np.sin(theta)
    return [dtheta_dt, dOmega_dt]

# Runge-Kutta method (RK4)
def rk4_step(f, t, u, dt, p):
    k1 = np.array(f(t, u, p))
    k2 = np.array(f(t + dt / 2, u + dt * k1 / 2, p))
    k3 = np.array(f(t + dt / 2, u + dt * k2 / 2, p))
    k4 = np.array(f(t + dt, u + dt * k3, p))
    return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Parameters
a = 0.1
l = 1.0
m = 1.0
g = 1.0
omega_values = [5, 10, 20]
initial_conditions = [np.pi / 4, 0]  # [theta(0), Omega(0)]
t_span = (0, 10)
dt = 0.01

# Solve the ODE for different omega values
for omega in omega_values:
    p = [a, l, m, g, omega]
    t_values = np.arange(t_span[0], t_span[1], dt)
    u_values = np.zeros((len(t_values), 2))
    u_values[0] = initial_conditions

    for i in range(1, len(t_values)):
        u_values[i] = rk4_step(equations, t_values[i-1], u_values[i-1], dt, p)

    theta_values = u_values[:, 0]

    # Plot the results
    plt.plot(t_values, theta_values, label=f'omega={omega}')

plt.xlabel('Time')
plt.ylabel('Theta')
plt.legend()
plt.title('Theta vs Time for different omega values')
plt.show()