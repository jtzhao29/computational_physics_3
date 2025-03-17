import numpy as np
import matplotlib.pyplot as plt


g = 10.0
gamma = 0.02
A = 0.02
omega = 2 * np.pi * 2
y0 = 0.3
v0 = 0.0

# 定义微分方程
def f(u, t, p):
    y, v = u
    dydt = v
    dvdt = -g - gamma * v
    return np.array([dydt, dvdt])


u0 = np.array([y0, v0])  
t0 = 0  
tf = 10  
dt = 0.01  

def runge_kutta_4(f, u0: np.ndarray, t0: float, tf: float, dt: float, p: dict) -> np.ndarray:
    t = t0
    u = u0
    length = int((tf - t0) / dt) + 1
    trajectory = np.zeros((length, 3))  
    time = 0
    v_before = u0[1]
    h_before = A * np.sin(omega * t0)
    
    while t < tf:
        h_now = A * np.sin(omega * t)
        v_now = A * omega * np.cos(omega * t)

        if (v_now - h_now) * (v_before - h_before) < 0:
            u[1] = 2 * A * omega * np.cos(omega * t) - u[1] 
        
        trajectory[time, :] = np.array([t, u[0], u[1]])
        
        k1 = dt * f(u, t, p)
        k2 = dt * f(u + 0.5 * k1, t + 0.5 * dt, p)
        k3 = dt * f(u + 0.5 * k2, t + 0.5 * dt, p)
        k4 = dt * f(u + k3, t + dt, p)
        
        u += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt
        time += 1
        
        v_before = u[1]
        h_before = h_now
    
    return trajectory

trajectory = runge_kutta_4(f, u0, t0, tf, dt, {})

tf2 = 990
trajectory2 = runge_kutta_4(f, u0, t0, tf2, dt, {})
u_990 = trajectory2[-1, 1:]
t_990 = trajectory2[-1, 0]

t02 = 999
tf2 = 1000
trajectory3 = runge_kutta_4(f, u_990, t02, tf2, dt, {})
h_trajectory3 = A * np.sin(omega * trajectory3[:, 0])


plt.figure(figsize=(10, 6))

plt.plot(trajectory[:, 0], trajectory[:, 1], label="乒乓球轨迹", color='blue')

h_trajectory = A * np.sin(omega * trajectory[:, 0])
plt.plot(trajectory[:, 0], h_trajectory, label="球拍轨迹", color='red', linestyle='--')

plt.xlabel("时间 (s)")
plt.ylabel("位置 (m)")
plt.title("乒乓球与球拍的运动轨迹")
plt.legend()

plt.savefig("figure/trajectory.png")

plt.show()


# 绘图
fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 改为 1 行 2 列

# 0-10s, y(t)
axs[0].plot(trajectory[:, 0], trajectory[:, 1], label="乒乓球轨迹", color='blue')
axs[0].plot(trajectory[:, 0], h_trajectory, label="球拍轨迹", color='red', linestyle='--')
axs[0].set_xlabel('Time (s)', fontsize=12)
axs[0].set_ylabel('y', fontsize=12)
axs[0].set_title('0-10s, y(t)', fontsize=14)
axs[0].legend()
axs[0].grid(True)

# 990-1000s, y(t)  注意x轴
axs[1].plot(trajectory3[:, 0], trajectory3[:, 1], label="乒乓球轨迹", color='blue')
axs[1].plot(trajectory3[:, 0], h_trajectory3, label="球拍轨迹", color='red', linestyle='--')
axs[0].set_ylabel('y', fontsize=12)
axs[1].set_title('990-1000s, y(t)', fontsize=14)
axs[1].legend()
axs[1].grid(True)


plt.tight_layout()
plt.show()