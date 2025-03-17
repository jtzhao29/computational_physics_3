import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# $$ \frac{d}{dt}y=v $$
# $$ \frac{d}{dt}v = -g-\gamma v$$

def f(u: np.ndarray, t: float, p: dict) -> np.ndarray:
    # 这个函数描述了方程组，输入输出都是array
    y, v = u
    g, gamma = p['g'], p['gamma']
    dy_dt = v
    dv_dt = -g - gamma * v
    return np.array([dy_dt, dv_dt])

# def bool_collision(u: np.array, t: float, p: dict) -> bool:
#     # 判断是否碰撞
#     y_before, v_before = u
#     A, omega = p['A'], p['omega']
    
#     # 计算当前时刻的状态
#     k1 = dt * f(u, t, p)
#     k2 = dt * f(u + 0.5 * k1, t + 0.5 * dt, p)
#     k3 = dt * f(u + 0.5 * k2, t + 0.5 * dt, p)
#     k4 = dt * f(u + k3, t + dt, p)
#     u += (k1 + 2 * k2 + 2 * k3 + k4) / 6
#     y_after, v_after = u

#     # 计算球拍的位置
#     racket_position_before = A * np.sin(omega * t)
#     racket_position_after = A * np.sin(omega * (t + dt))

#     # 判断球是否与球拍发生碰撞
#     if (y_before - racket_position_before) * (y_after - racket_position_after) < 0:
#         return True
#     else:
#         return False


def runge_kutta_4(f, u0: np.ndarray, t0: float, tf: float, dt: float, p: dict) -> np.ndarray:
    t = t0
    u = u0
    length = int((tf - t0) / dt) + 1
    trajectory = np.zeros((length, 3))
    time = 0
    g,gamma,A,omega = p['g'],p['gamma'],p['A'],p['omega']
    while t < tf:
        # 若碰撞，速度突变，否则直接带入龙格库塔法
        if abs(u[0]-A*np.sin(omega*t))<0.005 and u[1]<A*omega*np.cos(omega*t):
            u[1] = 2*A*omega*np.cos(omega*t)-u[1]
        y,v = u[0],u[1]
        trajectory[time,:] = np.array([t, y, v])
        k1 = dt * f(u, t, p)
        k2 = dt * f(u + 0.5 * k1, t + 0.5 * dt, p)
        k3 = dt * f(u + 0.5 * k2, t + 0.5 * dt, p)
        k4 = dt * f(u + k3, t + dt, p)
        u += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt
        time += 1
    return trajectory

params = {'g':10, 'gamma':0.02, 'A':0.02, 'omega':4*np.pi}
v0 = 0
dt = 0.001

# y0=0.3
y0=0.3
u0 = np.array([y0, v0])
t01 = 0
tf1 = 10
t02 = 990
tf2 = 1000

# trajectory1 = runge_kutta_4(f, u0, t01, tf2, dt, params)
# t1 = trajectory1[:int(10/dt), 0]
# y1 = trajectory1[:int(10/dt), 1]
# v1 = trajectory1[:int(10/dt), 2]

# t2 = trajectory1[int(990/dt):, 0]
# y2 = trajectory1[int(990/dt):, 1]
# v2 = trajectory1[int(990/dt):, 2]


# # # 展示0-10s的运动轨迹
# # trajectory1 = runge_kutta_4(f, u0, t01, tf1, dt, params)
# # t1 = trajectory1[:, 0]
# # y1 = trajectory1[:, 1]
# # v1 = trajectory1[:, 2]

# # # 展示990-1000s的运动轨迹
# # trajectory2 = runge_kutta_4(f, , t02, tf2, dt, params)
# # t2 = trajectory2[:, 0]
# # y2 = trajectory2[:, 1]
# # v2 = trajectory2[:, 2]

# # 展示0-10，990-1000s的轨迹（在同⼀张图也画出球拍的轨迹）

# fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# fig.suptitle(f'$y(t)$, when y0 = {y0}', fontsize=25)

# balcket1 = params['A']*np.sin(params['omega']*t1)
# balcket2 = params['A']*np.sin(params['omega']*t2)

# axs[0].plot(t1, y1, label='ball')
# axs[0].plot(t1, balcket1, label='racket', color='r')
# axs[0].set_ylabel('y', fontsize=25)
# axs[0].legend()

# axs[1].plot(t2, y2, label='ball', color='r')
# axs[1].plot(t2, balcket2, label='racket', color='b')
# axs[1].set_xlabel('Time [s]', fontsize=25)
# axs[1].set_ylabel('y', fontsize=25)
# axs[1].legend()

# axs[0].grid(True)
# axs[1].grid(True)

# plt.tight_layout()
# plt.savefig(f'figure/B_y0 = {y0}.png')
# plt.show()


# # 改变y0

def plot(y0,param,t01,tf2,dt):
    # 展示0-10s的运动轨迹
    trajectory1 = runge_kutta_4(f, u0, t01, tf2, dt, params)
    t1 = trajectory1[int(990/dt):, 0]
    y1 = trajectory1[int(990/dt):, 1]
    v1 = trajectory1[int(990/dt):, 2]
    balcket1 = params['A']*np.sin(params['omega']*t1)
    plt.plot(t1, y1, label='ball')
    plt.plot(t1, balcket1, label='racket', color='r')
    plt.ylabel('y', fontsize=25)
    plt.xlabel('Time [s]', fontsize=25)
    plt.title(f'$y(t)$, when y0 = {y0}', fontsize=25)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'figure/B_y0 = {y0}_ii.png')
    plt.show()

    plt.scatter(y1,v1)
    plt.xlabel('y', fontsize=25)    
    plt.ylabel('v', fontsize=25)
    plt.title(f'y vs v when y0 = {y0}', fontsize=25)
    plt.grid(True)
    plt.savefig(f'figure/B_y0 = {y0}_phase.png')
    plt.show()

# for y0 in [1,20,1]:
#     plot(y0,params,t01,tf2,dt)

plot(0.002,params,t01,tf2,dt)



# plt.scatter(theta_values,Omega_values)
# plt.xlabel("$\ theta $ ", fontsize=25)
# plt.ylabel("$\ Omega $", fontsize=25)
# plt.title(f"$\Omega$ vs $\Theta$ when $\omega$ = {p['omega']}", fontsize=25)
# plt.grid(True)
# plt.savefig(f'figure/A_{p["omega"]}_t={tf}_phase.png')
# plt.show()