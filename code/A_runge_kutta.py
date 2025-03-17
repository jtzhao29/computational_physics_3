import numpy as np
from typing import Callable, Union

def runge_kutta_4(f: Callable[[float, Union[float, np.ndarray]],
                               Union[float, np.ndarray]], 
                  y0: Union[float, np.ndarray], 
                  t0: float, 
                  tf: float, 
                  dt: float) -> list:
    """
    这个函数是用龙格库塔法求解方程的

    dt是时间步长

    输出的是一个列表包含每一步的时间和对应的 y 值 [(t1, y1), (t2, y2), ...]。
    
    """

    results = []
    t, y = t0, y0
    # 这里的df支持负的，就是对应于时间倒退的情况
    while (dt > 0 and t <= tf) or (dt < 0 and t >= tf):
        # 公式
        k1 = f(t, y)
        k2 = f(t + dt / 2, y + dt * k1 / 2)
        k3 = f(t + dt / 2, y + dt * k2 / 2)
        k4 = f(t + dt, y + dt * k3)

        # 更新 y 和 t
        y = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = t + dt

        # 保存结果
        results.append((t, y))

    return results
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