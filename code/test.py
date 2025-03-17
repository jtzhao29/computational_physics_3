import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

def plot_trajectory(file_path):
    df = pd.read_csv(file_path)
    t = df['t']
    theta = df['theta']
    Omega = df['Omega']

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(t, theta, label='Theta')
    axs[0].set_xlabel('Time (t)')
    axs[0].set_ylabel('Theta (θ)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t, Omega, label='Omega', color='r')
    axs[1].set_xlabel('Time (t)')
    axs[1].set_ylabel('Omega (Ω)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    
    plt.show()



# 获取data目录下所有json文件
data_dir = 'data'
files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

# 逐个文件读取并画图
for file in files:
    file_path = os.path.join(data_dir, file)
    plot_trajectory(file_path)