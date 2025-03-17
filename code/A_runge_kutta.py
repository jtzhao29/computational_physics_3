import numpy as np

def runge_kutta_4(f, y0, t0, tf, dt):
    """
    Solve an ODE using the 4th order Runge-Kutta method.

    Parameters:
    f : function
        The function that defines the ODE (dy/dt = f(t, y)).
    y0 : float or np.array
        The initial condition.
    t0 : float
        The initial time.
    tf : float
        The final time.
    dt : float
        The time step.

    Returns:
    t : np.array
        Array of time points.
    y : np.array
        Array of solution values at each time point.
    """
    t = np.arange(t0, tf + dt, dt)
    y = np.zeros((len(t), len(y0) if isinstance(y0, (list, np.ndarray)) else 1))
    y[0] = y0

    for i in range(1, len(t)):
        k1 = dt * f(t[i-1], y[i-1])
        k2 = dt * f(t[i-1] + dt/2, y[i-1] + k1/2)
        k3 = dt * f(t[i-1] + dt/2, y[i-1] + k2/2)
        k4 = dt * f(t[i-1] + dt, y[i-1] + k3)
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t, y

# Example usage:
if __name__ == "__main__":
    def f(t, y):
        return -2 * y + t

    y0 = 1
    t0 = 0
    tf = 10
    dt = 0.1

    t, y = runge_kutta_4(f, y0, t0, tf, dt)

    for ti, yi in zip(t, y):
        import matplotlib.pyplot as plt

        def pendulum_system(t, state):
            theta, omega = state
            dtheta_dt = omega
            domega_dt = (a / l) * omega**2 * np.cos(omega * t) * np.sin(theta) - (g / l) * np.sin(theta)
            return np.array([dtheta_dt, domega_dt])

        # Constants
        a = 1.0  # Example value for a
        l = 1.0  # Length of the pendulum
        g = 9.81  # Acceleration due to gravity

        # Initial conditions
        theta0 = np.pi / 4  # Initial angle (45 degrees)
        omega0 = 0.0  # Initial angular velocity
        y0 = np.array([theta0, omega0])

        # Time parameters
        t0 = 0
        tf = 10
        dt = 0.01

        # Solve the system using Runge-Kutta 4th order method
        t, y = runge_kutta_4(pendulum_system, y0, t0, tf, dt)

        # Plot the results
        plt.figure()
        plt.plot(t, y[:, 0], label='Theta (rad)')
        plt.plot(t, y[:, 1], label='Omega (rad/s)')
        plt.xlabel('Time (s)')
        plt.ylabel('Values')
        plt.legend()
        plt.title('Pendulum Motion')
        plt.show()