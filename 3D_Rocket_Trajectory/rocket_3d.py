import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

g0 = 9.80665
rho0 = 1.225
H = 8500.0

dry_mass = 2000.0
prop_mass = 8000.0
A = 3.0
Cd = 0.5
Isp = 300.0
thrust = 1.2e6

pitch_deg = 90.0
yaw_deg = 0.0
pitch_rad = np.deg2rad(pitch_deg)
yaw_rad = np.deg2rad(yaw_deg)

T_z = np.sin(pitch_rad)
T_x = np.cos(pitch_rad) * np.cos(yaw_rad)
T_y = np.cos(pitch_rad) * np.sin(yaw_rad)
thrust_direction = np.array([T_x, T_y, T_z])

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

print("="*70)
print("3D ROCKET TRAJECTORY SIMULATION")
print("="*70)
print(f"Total mass: {total_mass} kg")
print(f"Thrust: {thrust/1e6:.2f} MN")
print(f"TWR: {twr:.2f}")
print(f"Propellant: {prop_mass} kg")
print(f"Burn time: {burn_time:.1f} s")
print(f"Pitch: {pitch_deg:.1f} deg, Yaw: {yaw_deg:.1f} deg")
print(f"Thrust direction (x,y,z): ({T_x:.3f}, {T_y:.3f}, {T_z:.3f})")
print("="*70)

dt = 0.1
t_max = 1000.0
num_steps = int(t_max / dt) + 1

def air_density(h):
    if h < 0:
        h = 0.0
    return rho0 * np.exp(-h / H)

def derivatives(state, t):
    x, y, z, vx, vy, vz, m = state
    
    if m < dry_mass:
        m = dry_mass
    
    if m <= 0:
        raise RuntimeError(f"Mass became non-positive: m={m}")
    
    v_vec = np.array([vx, vy, vz])
    v = np.linalg.norm(v_vec)
    
    prop_remaining = m - dry_mass
    thrust_mag = 0.0
    if prop_remaining > 0.01 and t < burn_time:
        thrust_mag = thrust
    
    T_vec = thrust_mag * thrust_direction
    
    if v > 1e-8:
        rho = air_density(z)
        drag_mag = 0.5 * rho * Cd * A * v**2
        D_vec = -drag_mag * (v_vec / v)
    else:
        D_vec = np.array([0.0, 0.0, 0.0])
    
    g_vec = np.array([0.0, 0.0, -g0])
    a_vec = (T_vec + D_vec) / m + g_vec
    
    dm_dt = -thrust_mag / (Isp * g0) if (Isp > 0 and prop_remaining > 0.01 and t < burn_time) else 0.0
    
    return np.array([vx, vy, vz, a_vec[0], a_vec[1], a_vec[2], dm_dt])

def rk4_step(state, t, dt):
    k1 = derivatives(state, t)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivatives(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

t = 0.0
state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, total_mass])

times = []
xs, ys, zs = [], [], []
vxs, vys, vzs = [], [], []
ms = []
speeds = []

print("\nSimulation running in 3D...")
burn_end_time = burn_time
is_burning = True

for step in range(num_steps):
    times.append(t)
    xs.append(state[0])
    ys.append(state[1])
    zs.append(state[2])
    vxs.append(state[3])
    vys.append(state[4])
    vzs.append(state[5])
    ms.append(state[6])
    v = np.linalg.norm([state[3], state[4], state[5]])
    speeds.append(v)
    
    if is_burning and state[6] <= dry_mass + 10.0:
        burn_end_time = t
        is_burning = False
    
    if step > 100 and state[2] < 0:
        print(f"  Ground contact at t={t:.1f}s")
        break
    
    state = rk4_step(state, t, dt)
    t += dt
    
    if step % 200 == 0:
        print(f"  t={t:.1f}s, h={state[2]/1000:.2f}km, v={v:.0f}m/s, m={state[6]:.0f}kg")

print(f"Burn end: {burn_end_time:.1f}s")
print("Done\n")

times = np.array(times)
xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)
speeds = np.array(speeds)
ms = np.array(ms)

if len(times) == 0:
    raise RuntimeError("Simulation produced no data points")
burn_idx = np.argmin(np.abs(times - burn_end_time))

print("="*70)
print("RESULTS")
print("="*70)
print(f"\nAt BURN END ({burn_end_time:.1f}s):")
print(f"  Position: ({xs[burn_idx]/1000:.2f}, {ys[burn_idx]/1000:.2f}, {zs[burn_idx]/1000:.2f}) km")
print(f"  Height: {zs[burn_idx]/1000:.2f} km")
print(f"  Velocity: {speeds[burn_idx]:.0f} m/s")
print(f"  Mass: {ms[burn_idx]:.0f} kg")
print(f"\nMaximum values:")
if len(zs) > 0:
    print(f"  Max height: {np.max(zs)/1000:.2f} km (at ~{times[np.argmax(zs)]:.0f}s)")
if len(speeds) > 0:
    print(f"  Max velocity: {np.max(speeds):.0f} m/s")
if len(xs) > 0 and len(ys) > 0:
    print(f"  Max range (horizontal): {np.sqrt(np.max(xs)**2 + np.max(ys)**2)/1000:.2f} km")
print("="*70)

fig = plt.figure(figsize=(16, 10))

ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot(xs/1000, ys/1000, zs/1000, 'b-', linewidth=2, label='Trajectory')
ax1.scatter(xs[0]/1000, ys[0]/1000, zs[0]/1000, color='g', s=100, label='Start', marker='o')
ax1.scatter(xs[burn_idx]/1000, ys[burn_idx]/1000, zs[burn_idx]/1000, 
            color='r', s=100, label='Burn End', marker='X')
ax1.set_xlabel('X (km)')
ax1.set_ylabel('Y (km)')
ax1.set_zlabel('Z Height (km)')
ax1.set_title('3D Rocket Trajectory')
ax1.legend()

ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(xs/1000, zs/1000, 'g-', linewidth=2)
ax2.axhline(0, color='k', linestyle=':', alpha=0.3)
ax2.set_xlabel('X (km)')
ax2.set_ylabel('Z Height (km)')
ax2.set_title('Front View (X-Z)')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(ys/1000, zs/1000, 'm-', linewidth=2)
ax3.axhline(0, color='k', linestyle=':', alpha=0.3)
ax3.set_xlabel('Y (km)')
ax3.set_ylabel('Z Height (km)')
ax3.set_title('Side View (Y-Z)')
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(times, zs/1000, 'b-', linewidth=2, label='Height')
ax4.axvline(burn_end_time, color='r', linestyle='--', linewidth=2, label=f'Burn End')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Height (km)')
ax4.set_title('Height vs Time')
ax4.grid(True, alpha=0.3)
ax4.legend()

ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(times, speeds, 'r-', linewidth=2, label='Velocity')
ax5.axvline(burn_end_time, color='g', linestyle='--', linewidth=2)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Velocity (m/s)')
ax5.set_title('Velocity vs Time')
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(times, ms, 'm-', linewidth=2, label='Mass')
ax6.axvline(burn_end_time, color='r', linestyle='--', linewidth=2)
ax6.axhline(dry_mass, color='b', linestyle=':', linewidth=2, label='Dry Mass')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Mass (kg)')
ax6.set_title('Mass vs Time')
ax6.set_xlim([0, 50])
ax6.set_ylim([dry_mass - 100, total_mass + 200])
ax6.grid(True, alpha=0.3)
ax6.legend()

plt.tight_layout()
plt.savefig('rocket_3d_trajectory.png', dpi=150)
plt.show()

if PLOTLY_AVAILABLE:
    print("\nCreating interactive 3D visualization...")
    
    if len(xs) == 0 or len(zs) == 0:
        raise RuntimeError("Cannot create visualization: no trajectory data")
    max_x = np.max(np.abs(xs)) if len(xs) > 0 else 5
    max_z = np.max(np.abs(zs)) if len(zs) > 0 else 5
    max_range = max(max_x, max_z, 5) / 1000
    
    fig_plotly = go.Figure(data=go.Scatter3d(
        x=xs/1000,
        y=ys/1000,
        z=zs/1000,
        mode='lines+markers',
        marker=dict(
            size=2,
            color=speeds,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Velocity (m/s)")
        ),
        line=dict(color='blue', width=3),
        hovertemplate='X: %{x:.2f} km<br>Y: %{y:.2f} km<br>Z: %{z:.2f} km<extra></extra>',
        name='Trajectory'
    ))
    
    fig_plotly.add_scatter3d(x=[xs[0]/1000], y=[ys[0]/1000], z=[zs[0]/1000],
                             mode='markers', marker=dict(size=20, color='lime', symbol='circle'),
                             name='START: (0, 0, 0)')
    
    fig_plotly.add_scatter3d(x=[xs[burn_idx]/1000], y=[ys[burn_idx]/1000], z=[zs[burn_idx]/1000],
                             mode='markers', marker=dict(size=15, color='red', symbol='square'),
                             name=f'Burn End ({xs[burn_idx]/1000:.2f}, {ys[burn_idx]/1000:.2f}, {zs[burn_idx]/1000:.2f}) km')
    
    apogee_idx = np.argmax(zs)
    fig_plotly.add_scatter3d(x=[xs[apogee_idx]/1000], y=[ys[apogee_idx]/1000], z=[zs[apogee_idx]/1000],
                             mode='markers', marker=dict(size=15, color='orange', symbol='x'),
                             name=f'Apogee ({zs[apogee_idx]/1000:.2f} km)')
    
    fig_plotly.update_layout(
        title='3D Rocket Trajectory - Start at (0, 0, 0)',
        scene=dict(
            xaxis_title='X Position (km)',
            yaxis_title='Y Position (km)',
            zaxis_title='Z Position (km) - Height',
            aspectmode='data',
            bgcolor='lightgray',
            xaxis=dict(gridcolor='white', showspikes=False, zeroline=True, zerolinecolor='black'),
            yaxis=dict(gridcolor='white', showspikes=False, zeroline=True, zerolinecolor='black'),
            zaxis=dict(gridcolor='white', showspikes=False, zeroline=True, zerolinecolor='black'),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=50)
            )
        ),
        width=1200,
        height=900
    )
    
    fig_plotly.write_html('rocket_3d_interactive.html')
    print("Interactive visualization saved as: rocket_3d_interactive.html")
else:
    print("\nNote: Plotly not available. Skipping interactive 3D visualization.")
    print("Install plotly with: pip install plotly")
