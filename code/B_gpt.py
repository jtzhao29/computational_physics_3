import numpy as np
import matplotlib.pyplot as plt

# 参数 (与题目一致)
g = 10
gamma = 0.02
A = 0.02
omega = 2 * np.pi
y0 = 0.3
v0 = 0
dt = 0.001

def ball_ode(u, t, p):
    y, v = u
    dydt = v
    dvdt = -g - gamma  * v  
    return np.array([dydt, dvdt])

def racket_pos(t, p):
    return p['A'] * np.sin(p['omega'] * t)

def racket_vel(t, p):
    return p['A'] * p['omega'] * np.cos(p['omega'] * t)

def detect_collision(y_prev, y_curr, t_prev, t_curr, p):
    def equation_to_solve(t):
        y_ball = y_prev + (t - t_prev) * (y_curr - y_prev) / (t_curr - t_prev)
        y_racket = racket_pos(t, p)
        return y_ball - y_racket

    if equation_to_solve(t_prev) * equation_to_solve(t_curr) > 0:
        return False, None
    else:
        t_collision = (t_prev + t_curr) / 2
        while t_curr - t_prev > 1e-9:
            if equation_to_solve(t_prev) * equation_to_solve(t_collision) <= 0:
                t_curr = t_collision
            else:
                t_prev = t_collision
            t_collision = (t_prev + t_curr) / 2

        return True, t_collision

# 四阶 Runge-Kutta 方法 (同上一个版本)
def runge_kutta_4(f, u0, t0, tf, dt, p):
    t = t0
    u = u0
    trajectory = [np.array([t, u[0], u[1]])]

    while t < tf:
        y_prev, v_prev = u
        t_prev = t

        k1 = dt * f(u, t, p)
        k2 = dt * f(u + 0.5 * k1, t + 0.5 * dt, p)
        k3 = dt * f(u + 0.5 * k2, t + 0.5 * dt, p)
        k4 = dt * f(u + k3, t + dt, p)
        u = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt

        # 碰撞检测
        collided, t_collision = detect_collision(y_prev, u[0], t_prev, t, p)
        if collided:
            # 更新速度
            u[1] = 2 * racket_vel(t_collision, p) - u[1]
            t = t_collision

        trajectory.append(np.array([t, u[0], u[1]]))

    return np.array(trajectory)

# 参数
params = {'g': 10, 'gamma': 0.02, 'A': 0.02, 'omega': 2 * np.pi}
v0 = 0
y0 = 0.3
dt = 0.001

# 展示 0-10s 的运动轨迹
trajectory1 = runge_kutta_4(ball_ode, np.array([y0, v0]), 0, 10, dt, params)
t1 = trajectory1[:, 0]
y1 = trajectory1[:, 1]


# 获取 10 秒时的状态 (关键修改)
y_end = y1[-1]
v_end = trajectory1[-1, 2]
t_end = t1[-1] #其实就是10

# 展示 990-1000s 的运动轨迹 (使用 10 秒时的状态作为初始条件)
trajectory2 = runge_kutta_4(ball_ode, np.array([y_end, v_end]), t_end, 1000, dt, params)
t2 = trajectory2[:, 0]
y2 = trajectory2[:, 1]


# 绘图
fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 改为 1 行 2 列

# 0-10s, y(t)
axs[0].plot(t1, y1, label='Ball')
axs[0].plot(t1, racket_pos(t1, params), label='Racket', linestyle='--')
axs[0].set_xlabel('Time (s)', fontsize=12)
axs[0].set_ylabel('y', fontsize=12)
axs[0].set_title('0-10s, y(t)', fontsize=14)
axs[0].legend()
axs[0].grid(True)

# 990-1000s, y(t)  注意x轴
axs[1].plot(t2 - 990, y2, label='Ball')
axs[1].plot(t2 - 990, racket_pos(t2, params), label='Racket', linestyle='--')
axs[1].set_xlabel('Time (s)', fontsize=12)
axs[1].set_title('990-1000s, y(t)', fontsize=14)
axs[1].legend()
axs[1].grid(True)


plt.tight_layout()
plt.show()