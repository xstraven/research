import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant
M_sun = 1.989e30  # Mass of the Sun in kg
M_earth = 5.97e26  # Mass of the Earth in kg
AU = 1.496e11  # 1 Astronomical Unit in meters
day = 24 * 3600  # 1 day in seconds
year = 365.25 * day

# Time settings
dt = 6 * 3600  # Time step (6 hours)
total_time = 0.4 * year  # Total simulation time

# Initial conditions
r_earth = np.array([1 * AU, 0.0])  # Earth starts at 1 AU on x-axis
v_earth = np.array([0.0, 29.78e3])  # Earth's orbital velocity

r_comet = np.array([2 * AU, 1 * AU])  # Comet starts at (5AU, 4AU)
v_comet = np.array([-38000, 0])  # Initial velocity of comet


# def acceleration(r, mass, r_earth):
#     """Calculate acceleration due to gravity from Sun and Earth"""
#     r_sun = np.array([0, 0])
#     a_sun = -G * M_sun * (r - r_sun) / np.linalg.norm(r - r_sun) ** 3
#     a_earth = -G * M_earth * (r - r_earth) / np.linalg.norm(r - r_earth) ** 3
#     return a_sun + a_earth


def acceleration_by_sun(r):
    """Calculate acceleration due to gravity from Sun"""
    r_sun = np.array([0, 0])
    a_sun = -G * M_sun * (r - r_sun) / np.linalg.norm(r - r_sun) ** 3v
    """Calculate acceleration due to gravity from Earth"""
    a_earth = -G * M_earth * (r - r_earth) / np.linalg.norm(r - r_earth) ** 3
    return a_earth


def update(r_earth, v_earth, r_comet, v_comet):
    """Update positions and velocities"""
    a_earth = acceleration_by_sun(r_earth)  # Earth only affected by Sun
    a_comet = acceleration_by_sun(r_comet)  # Comet affected by Sun and Earth
    a_comet += acceleration_by_earth(r_comet, r_earth)
    r_earth_new = r_earth + v_earth * dt + 0.5 * a_earth * dt**2
    v_earth_new = v_earth + a_earth * dt

    r_comet_new = r_comet + v_comet * dt + 0.5 * a_comet * dt**2
    v_comet_new = v_comet + a_comet * dt

    return r_earth_new, v_earth_new, r_comet_new, v_comet_new


def simulate(r_earth, v_earth, r_comet, v_comet, num_steps):
    earth_trajectory = np.zeros((num_steps, 2))
    comet_trajectory = np.zeros((num_steps, 2))

    for i in range(num_steps):
        earth_trajectory[i] = r_earth
        comet_trajectory[i] = r_comet
        r_earth, v_earth, r_comet, v_comet = update(
            r_earth, v_earth, r_comet, v_comet
        )

    return earth_trajectory, comet_trajectory


# Run simulation
num_steps = int(total_time / dt)
earth_trajectory, comet_trajectory = simulate(
    r_earth, v_earth, r_comet, v_comet, num_steps
)

# Plot the orbits
plt.figure(figsize=(12, 12))
plt.plot(0, 0, "yo", markersize=20, label="Sun")
plt.plot(
    earth_trajectory[:, 0],
    earth_trajectory[:, 1],
    "b-",
    linewidth=0.5,
    label="Earth's orbit",
)
plt.plot(
    comet_trajectory[:, 0],
    comet_trajectory[:, 1],
    "r-",
    linewidth=0.5,
    label="Comet's orbit",
)

# Mark starting points
plt.plot(
    earth_trajectory[0, 0],
    earth_trajectory[0, 1],
    "bo",
    markersize=8,
    label="Earth start",
)
plt.plot(
    comet_trajectory[0, 0],
    comet_trajectory[0, 1],
    "ro",
    markersize=8,
    label="Comet start",
)

# Mark perihelion (closest approach to Sun)
perihelion_idx = np.argmin(np.sum(comet_trajectory**2, axis=1))
plt.plot(
    comet_trajectory[perihelion_idx, 0],
    comet_trajectory[perihelion_idx, 1],
    "mo",
    markersize=8,
    label="Comet perihelion",
)

# Add arrows to show comet's direction
arrow_indices = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4]
for i in arrow_indices:
    plt.arrow(
        comet_trajectory[i, 0],
        comet_trajectory[i, 1],
        comet_trajectory[i + 1, 0] - comet_trajectory[i, 0],
        comet_trajectory[i + 1, 1] - comet_trajectory[i, 1],
        head_width=0.1 * AU,
        head_length=0.15 * AU,
        fc="k",
        ec="k",
    )

plt.title("Comet's Orbit Influenced by Sun and Earth")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.legend()
plt.axis("equal")
plt.grid(True)

# Set axis limits
# max_range = max(np.max(np.abs(comet_trajectory)), 5 * AU)
# plt.xlim(-max_range, max_range)
# plt.ylim(-max_range, max_range)

plt.show()

# Print some orbital characteristics
distances_to_sun = np.linalg.norm(comet_trajectory, axis=1)
perihelion = np.min(distances_to_sun)
aphelion = np.max(distances_to_sun)

print(f"Comet's perihelion distance: {perihelion/AU:.2f} AU")
print(f"Comet's aphelion distance: {aphelion/AU:.2f} AU")
print(
    f"Comet's orbital eccentricity: {(aphelion - perihelion) / (aphelion + perihelion):.3f}"
)

# Calculate closest approach to Earth
distances_to_earth = np.linalg.norm(comet_trajectory - earth_trajectory, axis=1)
closest_approach = np.min(distances_to_earth)
closest_approach_idx = np.argmin(distances_to_earth)

print(f"Closest approach to Earth: {closest_approach/AU:.3f} AU")
print(f"Time of closest approach: {closest_approach_idx * dt / day:.1f} days")
