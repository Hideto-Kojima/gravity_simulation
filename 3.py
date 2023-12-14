import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def calculate_gravitational_force(mass1, mass2, position1, position2, G=6.67430e-11):
    r = position2 - position1
    distance = np.linalg.norm(r)
    magnitude = G * (mass1 * mass2) / distance**2
    force = magnitude * (r / distance)
    return force

def simulate_gravity(num_steps, dt, masses, initial_positions, initial_velocities):
    num_particles = len(masses)
    positions = np.zeros((num_steps, num_particles, 2))
    velocities = initial_velocities.copy()
    positions[0] = initial_positions

    for step in range(1, num_steps):
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

        velocities += total_forces * dt / masses[:, np.newaxis]
        positions[step] = positions[step - 1] + velocities * dt

    return positions

def update(frame, sc, lines, positions):
    sc.set_offsets(positions[frame])

    # 航跡の描画
    for i, line in enumerate(lines):
        line.set_data(positions[:frame, i, 0], positions[:frame, i, 1])

    return sc, *lines

if __name__ == "__main__":
    num_particles = 3
    num_steps = 1000
    dt = 0.1
    masses = np.array([1.0, 2.0, 3.0])
    initial_positions = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 3.0]
    ])
    initial_velocities = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0]
    ])

    particle_positions = simulate_gravity(num_steps, dt, masses, initial_positions, initial_velocities)

    fig, ax = plt.subplots()
    sc = ax.scatter([], [], marker='o', s=100, c='r', label='Particles')

    # 航跡の初期化
    lines = [ax.plot([], [], lw=1)[0] for _ in range(num_particles)]

    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Gravity Simulation')
    ax.legend()

    ani = FuncAnimation(fig, update, frames=num_steps, fargs=(sc, lines, particle_positions), interval=50, blit=True)
    plt.show()
