import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 前面第一问已经得到了方程组：$$\frac{d}{dt}\theta = \Omega$$

# $$\frac{d}{dt}\Omega = \frac{a}{l} \omega^2cos(\omega t )sin\theta -\frac{g}{l} sin(\theta)$$

def f(u: np.ndarray, t: float, p: dict) -> np.ndarray:
    # 这个函数描述了方程组，输入输出都是array
    theta, Omega_1 = u
    a, l, m, g, omega = p['a'], p['l'], p['m'], p['g'], p['omega']
    dtheta_dt = Omega_1
    dOmega_dt = (a / l) * omega**2 * np.cos(omega * t) * np.sin(theta) - (g / l) * np.sin(theta)
    return np.array([dtheta_dt, dOmega_dt])

def runge_kutta_4(f, u0: np.ndarray, t0: float, tf: float, dt: float, p: dict) -> np.ndarray:
    t = t0
    u = u0
    length = int((tf - t0) / dt) + 1
    trajectory = np.zeros((length, 3))
    time = 0
    while t < tf:
        theta,omega_1 = u[0],u[1]
        trajectory[time,:] = np.array([t, theta, omega_1])
        k1 = dt * f(u, t, p)
        k2 = dt * f(u + 0.5 * k1, t + 0.5 * dt, p)
        k3 = dt * f(u + 0.5 * k2, t + 0.5 * dt, p)
        k4 = dt * f(u + k3, t + dt, p)
        u += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt
        time += 1
    return trajectory

# 指定初始状态
p = {'a': 0.1, 'l': 1.0, 'm': 1.0, 'g': 1.0, 'omega': 2*np.pi}
u0 = np.array([np.pi / 5*4, 0.0])
t0 = 0.0
tf = 50
dt = 0.01

trajectory = runge_kutta_4(f, u0, t0, tf, dt, p)
# 保存数据
df = pd.DataFrame(trajectory, columns=['t', 'theta', 'Omega'])
df.to_csv(f'data/A_{p["omega"]}.json', index=False)


t_values = trajectory[:, 0]
theta_values = trajectory[:, 1]
Omega_values = trajectory[:, 2]

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

fig.suptitle(f'$\\theta(t)$ and $\\dot{{\\theta}}(t)$, when $\\omega$ = {p["omega"]}', fontsize=25)

axs[0].plot(t_values, theta_values, label='$\\theta(t)$')
axs[0].set_ylabel('θ [rad]', fontsize=25)
axs[0].legend()

axs[1].plot(t_values, Omega_values, label='$\\dot{\\theta}(t)$', color='r')
axs[1].set_xlabel('Time [s]', fontsize=25)
axs[1].set_ylabel('Omega (Ω)', fontsize=25)
axs[1].legend()

axs[0].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'figure/A_{p["omega"]}_t={tf}.png')
plt.show()



plt.scatter(theta_values,Omega_values)
plt.xlabel("$\ theta $ ", fontsize=25)
plt.ylabel("$\ Omega $", fontsize=25)
plt.title(f"$\Omega$ vs $\Theta$ when $\omega$ = {p['omega']}", fontsize=25)
plt.grid(True)
plt.savefig(f'figure/A_{p["omega"]}_t={tf}_phase.png')
plt.show()