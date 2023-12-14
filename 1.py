import numpy as np
import matplotlib.pyplot as plt

def calculate_gravitational_force(mass1, mass2, position1, position2, G=6.67430e-11):
    """
    2つの質点間の重力を計算する関数

    Parameters:
    - mass1 (float): 質点1の質量
    - mass2 (float): 質点2の質量
    - position1 (np.ndarray): 質点1の位置 (x, y)
    - position2 (np.ndarray): 質点2の位置 (x, y)
    - G (float): 重力定数 (デフォルトは地球の場合の値)

    Returns:
    - np.ndarray: 質点1に働く重力のベクトル (x, y)
    """
    # 質点1から質点2へのベクトル
    r = position2 - position1
    # 距離の絶対値
    distance = np.linalg.norm(r)
    # 万有引力の法則に基づく重力の大きさ
    magnitude = G * (mass1 * mass2) / distance**2
    # 重力のベクトル
    force = magnitude * (r / distance)
    return force

def simulate_gravity(num_steps, dt, masses, initial_positions, initial_velocities):
    """
    重力多体問題のシミュレーションを行う関数

    Parameters:
    - num_steps (int): シミュレーションステップ数
    - dt (float): タイムステップ
    - masses (np.ndarray): 質点の質量の配列
    - initial_positions (np.ndarray): 質点の初期位置の配列 (N行2列)
    - initial_velocities (np.ndarray): 質点の初期速度の配列 (N行2列)

    Returns:
    - np.ndarray: シミュレーション結果の質点の位置 (N行2列，各行が各質点の位置)
    """
    num_particles = len(masses)
    positions = np.zeros((num_steps, num_particles, 2))
    velocities = initial_velocities.copy()
    positions[0] = initial_positions

    for step in range(1, num_steps):
        # 質点同士の重力を計算
        total_forces = np.zeros((num_particles, 2))
        for i in range(num_particles):
            for j in range(num_particles):
                if i != j:
                    force = calculate_gravitational_force(
                        masses[i],
                        masses[j],
                        positions[step - 1, i],
                        positions[step - 1, j]
                    )
                    total_forces[i] += force

        # 速度の更新
        velocities += total_forces * dt / masses[:, np.newaxis]

        # 位置の更新
        positions[step] = positions[step - 1] + velocities * dt

    return positions

if __name__ == "__main__":
    # シミュレーションパラメータ
    num_particles = 3
    num_steps = 1000
    dt = 0.1

    # 質点の質量
    masses = np.array([1.0, 2.0, 3.0])

    # 質点の初期位置
    initial_positions = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 3.0]
    ])

    # 質点の初期速度
    initial_velocities = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0]
    ])

    # シミュレーション実行
    particle_positions = simulate_gravity(num_steps, dt, masses, initial_positions, initial_velocities)

    # シミュレーション結果の可視化
    for i in range(num_particles):
        plt.plot(particle_positions[:, i, 0], particle_positions[:, i, 1], label=f'Particle {i+1}')

    plt.title('Gravity Simulation')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.show()
