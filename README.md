# 计算物理导论-Himework 3: 常微分方程
## 题目描述
## 项目概况
## 结果及分析
### A. Kapitza摆
#### 1.求出系统的运动⽅程

以 $\theta$,$\dot{\theta}$为广义坐标和广义速度：
得到小球的动能：
$$T = \frac{1}{2}m[(-a \omega sin\omega t+l\dot{\theta}sin \theta)^2+(l \dot{\theta} cos \theta)^2]$$
$$T  = \frac{m}{2}[(l \dot{\theta} )^2+(a \omega sin(\omega t))^2-2a \omega l \dot{\theta}sin(\omega t)sin \theta]$$
势能(以x轴为势能零点)：
$$V = mg(a cos\omega t -lcos \theta)$$
体系的拉格朗日量为：
$$L = T-V$$
$$L = T - V =\frac{m}{2}[(l \dot{\theta} )^2+(a \omega sin(\omega t))^2-2a \omega l \dot{\theta}sin(\omega t)sin \theta] - mg  ( a cos(ωt) - l  cos(θ))$$
由拉格朗日方程：
$$\frac{d}{dt}\frac{\partial L}{\partial \dot{\theta}} - \frac{\partial L}{\partial \theta} = 0$$
化简得到：
$$l\ddot{\theta} = a \omega^2cos(\omega t )sin\theta -gsin(\theta)$$
不妨将$\dot{\theta}$记作$\Omega$，写成题目要求的形式：
$$\frac{d}{dt}\theta = \Omega$$

$$\frac{d}{dt}\Omega = \frac{a}{l} \omega^2cos(\omega t )sin\theta -\frac{g}{l} sin(\theta)$$

#### 2. 写出Runge-Kutta求解程序

在写出程序之前，思考以下问题：
- Q:任何⼀个别的ODE问题也可以抽象成$f(u,t,p)$吗？
- A:不一定，能写出$f(u,t,p)$形式的方程都是$\frac{d}{dt}u$在原方程中能够写出解析解的
- Q：（1）中显式写出动力系统的方程的意义：
- A：因为龙格库塔法只能解决一阶常微分方程，所以只有把$\ddot{\theta}$写成$\dot{\Omega}$的形式，以增加未知数个数的代价换取方程组由二阶将为一阶
- 此外，本程序使用使⽤精度较⾼的格式，例如课程介绍的````RK4 ````
得到求解的程序如下：
````python
def runge_kutta_4(f, u0: np.ndarray, t0: float, tf: float, dt: float, p: dict) -> np.ndarray:
    t = t0
    u = u0
    length = int((tf - t0) / dt) + 1
    trajectory = np.zeros((length, 3))
    time = 0
    while t < tf+dt:
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
````

这个函数是在已知t时的值后求解t+dt时的值。

#### 3. 得到实验结果
取$l=m=g=1$,$a = 0.1$,$\omega = 5,10,20$，分别画出$θ(t)$和$ω(t)$的图像。得到结果如下：
$\omega = 5$:
![alt text](figure/A_5.0.png)
$\omega = 10$:
![alt text](figure/A_10.0.png)
$\omega = 20$:
![alt text](figure/A_20.0.png)
由上述结果我发现：
1. 当$\omega = 5$时，系统的振荡没有周期性
2. 当$\omega = 10，20$时，系统的振荡表现出了很强的周期性
   
为了进一步验证该发现，做出$ \theta$,$\omega$的相图如下：
$\omega = 5$:
![alt text](figure/A_5.0_t=10000_phase.png)
$\omega = 10$:
![alt text](figure/A_10.0_t=1000_phase.png)
$\omega = 20$:
![alt text](figure/A_20.0_t=1000_phase.png)

#### 4.理论解释

### B. 乒乓球


