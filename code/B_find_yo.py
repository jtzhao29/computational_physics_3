import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks

# 参数
g = 10
gamma = 0.02
A = 0.02
omega = 2 * np.pi
dt = 0.001

# 微分方程
def ball_ode(u, t, p):
    y, v = u
    dydt = v
    dvdt = -g - (gamma / 1) * v
    return np.array([dydt, dvdt])

# 球拍位置
def racket_pos(t, p):
    return p['A'] * np.sin(p['omega'] * t)

# 球拍速度
def racket_vel(t, p):
    return p['A'] * p['omega'] * np.cos(p['omega'] * t)

# 碰撞检测 (修正)
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

# 四阶 Runge-Kutta 方法
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

        collided, t_collision = detect_collision(y_prev, u[0], t_prev, t, p)
        if collided:
            u[1] = 2 * racket_vel(t_collision, p) - u[1]
            t = t_collision

        trajectory.append(np.array([t, u[0], u[1]]))

    return np.array(trajectory)

# 寻找分岔 (以 y0 为分岔参数)
def find_bifurcation(y0_values, params, t_span, dt):
    bifurcation_data = []

    for y0 in y0_values:
        # 使用长时间模拟的最后状态作为初始条件, 避免重复计算
        trajectory = runge_kutta_4(ball_ode, np.array([y0, 0]), 0, t_span[0], dt, params)
        u0_long = trajectory[-1, 1:]  # 获取长时间模拟结束时的状态
        trajectory = runge_kutta_4(ball_ode, u0_long, t_span[0], t_span[1], dt, params)

        t = trajectory[:, 0]
        y = trajectory[:, 1]
        v = trajectory[:,2]

        # 方法1: 查找峰值 (y 的局部最大值)
        peaks, _ = find_peaks(y, height=0)  # 找到 y 值的所有峰值
        peak_times = t[peaks]
        peak_values = y[peaks]

        # 方法2:  Poincare 截面 (每隔一个驱动周期记录一次 y 值)
        #poincare_times = np.arange(t_span[0], t_span[1], 2*np.pi/params['omega'])
        #poincare_values = np.array([trajectory[(np.abs(trajectory[:, 0] - poincare_time)).argmin()][1] for poincare_time in poincare_times])

        #记录
       # for val in poincare_values:
       #      bifurcation_data.append([y0, val])
        for val in peak_values:
            bifurcation_data.append([y0, val])

    return np.array(bifurcation_data)

# 参数
params = {'g': 10, 'gamma': 0.02, 'A': 0.02, 'omega': 2 * np.pi}


# 不同的 y0 值
y0_values = np.linspace(0.2, 1.0, 200)  # 更密的采样

# 模拟时间
t_span = (500,600) # 先进行长时间模拟，再分析稳态
dt = 0.001

# 执行分岔分析
bifurcation_data = find_bifurcation(y0_values, params, t_span, dt)

# 绘制分岔图
plt.figure(figsize=(10, 6))
plt.scatter(bifurcation_data[:, 0], bifurcation_data[:, 1], s=1, marker='.', c='blue')  # 用小点绘制
plt.xlabel('y0', fontsize=15)
plt.ylabel('y (peak values)', fontsize=15)
plt.title('Bifurcation Diagram (y0)', fontsize=18)
plt.grid(True)
plt.show()