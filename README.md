# 计算物理导论-Himework 3: 常微分方程
## 题目描述
## 项目概况
## 结果及分析
### A. Kapitza摆
#### 1.求出系统的运动⽅程

以$\theta $,$ \dot{\theta} $为广义坐标和广义速度：
得到小球的动能：
$$ T = \frac{1}{2}m[(-a \omega sin\omega t+l\dot{\theta}sin \theta)^2+(l \dot{\theta} cos \theta)^2] $$
$$ T  = \frac{m}{2}[(l \dot{\theta} )^2+(a \omega sin(\omega t))^2-2a \omega l \dot{\theta}sin(\omega t)sin \theta] $$
势能(以x轴为势能零点)：
$$ V = mg(a cos\omega t -lcos \theta) $$
体系的拉格朗日量为：
$$ L = T-V$$
$$ L = T - V =\frac{m}{2}[(l \dot{\theta} )^2+(a \omega sin(\omega t))^2-2a \omega l \dot{\theta}sin(\omega t)sin \theta] - mg  ( a cos(ωt) - l  cos(θ)) $$
由拉格朗日方程：
$$ \frac{d}{dt}\frac{\partial L}{\partial \dot{\theta}} - \frac{\partial L}{\partial \theta} = 0 $$
化简得到：
$$ l\ddot{\theta} = a \omega^2cos(\omega t )sin\theta -gsin(\theta) $$
不妨将$\dot{\theta}$记作$\Omega$，写成题目要求的形式：
$$ \frac{d}{dt}\theta = \Omega$$

$$ \frac{d}{dt}\Omega = \frac{a}{l} \omega^2cos(\omega t )sin\theta -\frac{g}{l} sin(\theta) $$






