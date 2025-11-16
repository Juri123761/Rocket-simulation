import numpy as np
import matplotlib.pyplot as plt

g0 = 9.80665
rho0 = 1.225
H = 8500.0

dry_mass = 2000.0
prop_mass = 8000.0
A = 3.0
Cd = 0.5
Isp = 300.0
thrust = 1.2e6
pitch_angle_deg = 85.0
pitch_angle_rad = np.deg2rad(pitch_angle_deg)

if Isp <= 0:
    raise ValueError("Isp must be positive")
if thrust <= 0:
    raise ValueError("Thrust must be positive")
mass_flow_rate = thrust / (Isp * g0)
if mass_flow_rate <= 0:
    raise ValueError("Mass flow rate must be positive")
burn_time = prop_mass / mass_flow_rate

total_mass = dry_mass + prop_mass
twr = thrust / (total_mass * g0)

print("="*60)
print("ROCKET PARAMETERS")
print("="*60)
print(f"Total mass: {total_mass} kg")
print(f"Thrust: {thrust/1e6:.2f} MN")
print(f"TWR: {twr:.2f}")
print(f"Propellant: {prop_mass} kg")
print(f"Mass flow: {mass_flow_rate:.1f} kg/s")
print(f"Burn time: {burn_time:.1f} s")
print("="*60)

dt = 0.1
t_max = 1000.0
num_steps = int(t_max / dt) + 1

def air_density(h):
    if h < 0:
        h = 0.0
    return rho0 * np.exp(-h / H)

def derivatives(state, t):
    x, y, vx, vy, m = state
    
    if m < dry_mass:
        m = dry_mass
    
    if m <= 0:
        raise RuntimeError(f"Mass became non-positive: m={m}")
    
    v_vec = np.array([vx, vy])
    v = np.linalg.norm(v_vec)
    
    prop_remaining = m - dry_mass
    thrust_mag = 0.0
    if prop_remaining > 0.01 and t < burn_time:
        thrust_mag = thrust
    
    T_vec = thrust_mag * np.array([np.cos(pitch_angle_rad), np.sin(pitch_angle_rad)])
    
    if v > 1e-8:
        rho = air_density(y)
        drag_mag = 0.5 * rho * Cd * A * v**2
        D_vec = -drag_mag * (v_vec / v)
    else:
        D_vec = np.array([0.0, 0.0])
    
    a_vec = (T_vec + D_vec) / m + np.array([0.0, -g0])
    dm_dt = -thrust_mag / (Isp * g0) if (Isp > 0 and prop_remaining > 0.01 and t < burn_time) else 0.0
    
    return np.array([vx, vy, a_vec[0], a_vec[1], dm_dt])

def rk4_step(state, t, dt):
    k1 = derivatives(state, t)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivatives(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

t = 0.0
state = np.array([0.0, 0.0, 0.0, 0.0, total_mass])

times, xs, ys, vxs, vys, ms = [], [], [], [], [], []

print("\nSimulation running...")
burn_end_time = burn_time
is_burning = True

for step in range(num_steps):
    times.append(t)
    xs.append(state[0])
    ys.append(state[1])
    vxs.append(state[2])
    vys.append(state[3])
    ms.append(state[4])
    
    if is_burning and state[4] <= dry_mass + 10.0:
        burn_end_time = t
        is_burning = False
    
    if step > 100 and state[1] < 0:
        print(f"  Ground contact at t={t:.1f}s")
        break
    
    state = rk4_step(state, t, dt)
    t += dt

print(f"Burn end: {burn_end_time:.1f}s (planned: {burn_time:.1f}s)")
print("Done\n")

times = np.array(times)
xs = np.array(xs)
ys = np.array(ys)
vxs = np.array(vxs)
vys = np.array(vys)
speeds = np.sqrt(vxs**2 + vys**2)
ms = np.array(ms)

if len(times) == 0:
    raise RuntimeError("Simulation produced no data points")
burn_idx = np.argmin(np.abs(times - burn_end_time))
height_at_burn_end = ys[burn_idx]
velocity_at_burn_end = speeds[burn_idx]
mass_at_burn_end = ms[burn_idx]

print("="*60)
print("RESULTS")
print("="*60)
print(f"\nAt BURN END ({burn_end_time:.1f}s):")
print(f"  Height: {height_at_burn_end/1000:.2f} km")
print(f"  Velocity: {velocity_at_burn_end:.0f} m/s ({velocity_at_burn_end*3.6:.0f} km/h)")
print(f"  Mass: {mass_at_burn_end:.0f} kg")
print(f"\nMaximum values:")
if len(ys) > 0:
    print(f"  Max height: {np.max(ys)/1000:.2f} km (at ~{times[np.argmax(ys)]:.0f}s)")
if len(speeds) > 0:
    print(f"  Max velocity: {np.max(speeds):.0f} m/s ({np.max(speeds)*3.6:.0f} km/h)")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(xs/1000.0, ys/1000.0, 'b-', linewidth=2)
axes[0,0].set_xlabel("Horizontal (km)")
axes[0,0].set_ylabel("Height (km)")
axes[0,0].set_title("Rocket Trajectory (2D)")
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(times, ys/1000, 'g-', linewidth=2)
axes[0,1].axvline(burn_end_time, color='r', linestyle='--', linewidth=2, label=f'Burn End ({burn_end_time:.1f}s)')
axes[0,1].plot(burn_end_time, height_at_burn_end/1000, 'ro', markersize=10, label=f'h={height_at_burn_end/1000:.1f} km')
axes[0,1].plot(times[np.argmax(ys)], np.max(ys)/1000, 'y*', markersize=15, label=f'Apogee h={np.max(ys)/1000:.1f} km')
axes[0,1].set_xlabel("Time (s)")
axes[0,1].set_ylabel("Height (km)")
axes[0,1].set_title("Height vs Time")
axes[0,1].grid(True, alpha=0.3)
axes[0,1].legend()

axes[1,0].plot(times, speeds, 'r-', linewidth=2)
axes[1,0].axvline(burn_end_time, color='g', linestyle='--', linewidth=2, label=f'Burn End ({burn_end_time:.1f}s)')
axes[1,0].plot(burn_end_time, velocity_at_burn_end, 'go', markersize=10, label=f'v={velocity_at_burn_end:.0f} m/s')
axes[1,0].set_xlabel("Time (s)")
axes[1,0].set_ylabel("Velocity (m/s)")
axes[1,0].set_title("Velocity vs Time")
axes[1,0].grid(True, alpha=0.3)
axes[1,0].legend()

axes[1,1].plot(times, ms, 'm-', linewidth=2, label='Mass')
axes[1,1].axvline(burn_end_time, color='r', linestyle='--', linewidth=2, label=f'Burn End ({burn_end_time:.1f}s)')
axes[1,1].axhline(dry_mass, color='b', linestyle=':', linewidth=2, label='Dry Mass')
axes[1,1].set_xlabel("Time (s)")
axes[1,1].set_ylabel("Mass (kg)")
axes[1,1].set_title("Mass vs Time")
axes[1,1].grid(True, alpha=0.3)
axes[1,1].legend()
axes[1,1].set_xlim([0, 50])
axes[1,1].set_ylim([dry_mass - 100, total_mass + 200])

plt.tight_layout()
plt.savefig('trajectory.png', dpi=150)
plt.show()
