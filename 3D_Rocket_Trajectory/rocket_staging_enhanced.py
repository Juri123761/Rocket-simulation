import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

g0 = 9.80665
R_earth = 6.371e6
rho0 = 1.225
T0 = 288.15
p0 = 101325.0
R_gas = 287.05

stages = [
    {'name': 'Stage 1', 'dry_mass': 15000.0, 'prop_mass': 40000.0, 
     'thrust': 5e6, 'Isp': 280.0, 'A': 3.0, 'Cd': 0.5},
    {'name': 'Stage 2', 'dry_mass': 3000.0, 'prop_mass': 10000.0, 
     'thrust': 1.5e6, 'Isp': 300.0, 'A': 2.0, 'Cd': 0.4},
    {'name': 'Stage 3', 'dry_mass': 500.0, 'prop_mass': 2000.0, 
     'thrust': 400000.0, 'Isp': 320.0, 'A': 1.0, 'Cd': 0.35}
]

payload_mass = 200.0
pitch_deg_stage = [90.0, 85.0, 80.0]
yaw_deg = 0.0

cumulative_times = [0.0]
stage_min_masses = []
for i, stage in enumerate(stages):
    if stage['Isp'] <= 0:
        raise ValueError(f"Isp must be positive for {stage['name']}")
    if stage['thrust'] <= 0:
        raise ValueError(f"Thrust must be positive for {stage['name']}")
    mass_flow_rate = stage['thrust'] / (stage['Isp'] * g0)
    if mass_flow_rate <= 0:
        raise ValueError(f"Mass flow rate must be positive for {stage['name']}")
    stage['burn_time'] = stage['prop_mass'] / mass_flow_rate
    stage['mass_flow_rate'] = mass_flow_rate
    cumulative_times.append(cumulative_times[-1] + stage['burn_time'])
    min_mass = payload_mass + sum([stages[j]['dry_mass'] for j in range(i, len(stages))])
    stage_min_masses.append(min_mass)

stage_masses_after = []
remaining = payload_mass
for i in range(len(stages) - 1, -1, -1):
    remaining += stages[i]['dry_mass'] + stages[i]['prop_mass']
    stage_masses_after.insert(0, remaining)

dt = 0.05
t_max = 300.0
num_steps = int(t_max / dt) + 1

def gravitational_acceleration(h):
    if h < 0:
        h = 0.0
    return g0 * (R_earth / (R_earth + h))**2

def wind_velocity(h):
    if h < 0:
        h = 0.0
    
    jetstream_height = 10000.0
    jetstream_max = 120.0
    
    if h < 8000:
        wind_speed = jetstream_max * 0.3 * (h / 8000)
    elif h < 12000:
        wind_speed = jetstream_max * (0.3 + 0.7 * (h - 8000) / 4000)
    elif h < 20000:
        wind_speed = jetstream_max * np.exp(-(h - 12000) / 8000)
    else:
        wind_speed = 5.0
    
    return np.array([wind_speed, 0.0, 0.0])

_ISA_T_h32 = 216.65 + 0.001 * 12000
_ISA_p_h20_at_32 = p0 * (216.65 / T0)**5.256 * np.exp(-9000 / 6341.62)
_ISA_p_h32 = _ISA_p_h20_at_32 * (_ISA_T_h32 / 216.65)**(-g0 / (0.001 * R_gas))
_ISA_p_h47 = _ISA_p_h32 * np.exp(-15000 / 7000)

def ISA_atmosphere(h):
    if h < 0:
        h = 0.0
    
    T = T0
    p = p0
    
    if h < 11000:
        T = T0 - 0.0065 * h
        p = p0 * (T / T0)**5.256
    elif h < 20000:
        T = 216.65
        p_h11 = p0 * (216.65 / T0)**5.256
        p = p_h11 * np.exp(-(h - 11000) / 6341.62)
    elif h < 32000:
        T = 216.65 + 0.001 * (h - 20000)
        p_h20 = p0 * (216.65 / T0)**5.256 * np.exp(-9000 / 6341.62)
        p = p_h20 * (T / 216.65)**(-g0 / (0.001 * R_gas))
    elif h < 47000:
        T = 228.65
        p = _ISA_p_h32 * np.exp(-(h - 32000) / 7000)
    else:
        T = 228.65 + 0.0028 * (h - 47000)
        p = _ISA_p_h47 * np.exp(-(h - 47000) / 6000)
    
    rho = p / (R_gas * T)
    return max(rho, 1e-10), T, p

def air_density(h):
    rho, _, _ = ISA_atmosphere(h)
    return rho

def get_current_stage(t):
    for i in range(len(stages)):
        if t < cumulative_times[i+1]:
            return i, t - cumulative_times[i]
    return len(stages) - 1, t - cumulative_times[-1]

def derivatives(state, t, dt_step):
    x, y, z, vx, vy, vz, m = state
    
    stage_idx, time_in_stage = get_current_stage(t)
    stage = stages[stage_idx]
    min_mass = stage_min_masses[stage_idx]
    
    if m < min_mass:
        m = min_mass
    
    if m <= 0:
        raise RuntimeError(f"Mass became non-positive: m={m} at t={t:.2f}s, stage={stage_idx}")
    
    v_vec = state[3:6]
    v = np.linalg.norm(v_vec)
    
    thrust_mag = 0.0
    prop_remaining = m - min_mass
    if prop_remaining > 0.01 and 0 <= time_in_stage < stage['burn_time']:
        thrust_mag = stage['thrust']
    
    pitch_rad = np.deg2rad(pitch_deg_stage[stage_idx])
    yaw_rad = np.deg2rad(yaw_deg)
    T_z = np.sin(pitch_rad)
    T_x = np.cos(pitch_rad) * np.cos(yaw_rad)
    T_y = np.cos(pitch_rad) * np.sin(yaw_rad)
    T_vec = thrust_mag * np.array([T_x, T_y, T_z])
    
    wind_vec = wind_velocity(z)
    v_rel = v_vec - wind_vec
    v_rel_mag = np.linalg.norm(v_rel)
    
    if v_rel_mag > 1e-8:
        rho = air_density(z)
        drag_mag = 0.5 * rho * stage['Cd'] * stage['A'] * v_rel_mag**2
        D_vec = -drag_mag * (v_rel / v_rel_mag)
    else:
        D_vec = np.zeros(3)
    
    g = gravitational_acceleration(z)
    a_vec = (T_vec + D_vec) / m + np.array([0.0, 0.0, -g])
    
    if thrust_mag > 0 and stage['Isp'] > 0:
        dm_dt = -thrust_mag / (stage['Isp'] * g0)
        if dt_step > 0 and m + dm_dt * dt_step < min_mass:
            dm_dt = (min_mass - m) / dt_step
    else:
        dm_dt = 0.0
    
    return np.array([vx, vy, vz, a_vec[0], a_vec[1], a_vec[2], dm_dt])

def rk4_step(state, t, dt):
    k1 = derivatives(state, t, dt)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, dt)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, dt)
    k4 = derivatives(state + dt * k3, t + dt, dt)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

total_mass = sum([s['dry_mass'] + s['prop_mass'] for s in stages]) + payload_mass

print("="*80)
print("ENHANCED 3D ROCKET TRAJECTORY WITH STAGING")
print("="*80)
print(f"Total mass (start): {total_mass:.0f} kg")
print(f"Payload: {payload_mass:.0f} kg")
print(f"Number of stages: {len(stages)}")
print("\nStage Details:")
for i, stage in enumerate(stages):
    twr = stage['thrust'] / (total_mass * g0)
    print(f"  {stage['name']}:")
    print(f"    Dry mass: {stage['dry_mass']:.0f} kg")
    print(f"    Propellant: {stage['prop_mass']:.0f} kg")
    print(f"    Thrust: {stage['thrust']/1e6:.2f} MN")
    print(f"    Isp: {stage['Isp']:.0f} s")
    print(f"    Burn time: {stage['burn_time']:.1f} s")
    print(f"    TWR: {twr:.2f}")
print("\nPhysical effects:")
print("  Height-dependent gravity g(y) = g0 * (R/(R+y))^2")
print("  Wind model with jetstream (max 120 m/s at 10km)")
print("  ISA Standard Atmosphere (complete)")
print("  Relative velocity for drag")
print("="*80)

t = 0.0
state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, total_mass])

times = np.zeros(num_steps)
xs = np.zeros(num_steps)
ys = np.zeros(num_steps)
zs = np.zeros(num_steps)
vxs = np.zeros(num_steps)
vys = np.zeros(num_steps)
vzs = np.zeros(num_steps)
ms = np.zeros(num_steps)
speeds = np.zeros(num_steps)
stage_info = np.zeros(num_steps, dtype=int)
gs = np.zeros(num_steps)
winds = np.zeros(num_steps)
densities = np.zeros(num_steps)

print("\nSimulation running...")

next_separation_idx = 0
actual_steps = num_steps

for step in range(num_steps):
    times[step] = t
    xs[step] = state[0]
    ys[step] = state[1]
    zs[step] = state[2]
    vxs[step] = state[3]
    vys[step] = state[4]
    vzs[step] = state[5]
    ms[step] = state[6]
    
    v_vec = state[3:6]
    v = np.linalg.norm(v_vec)
    speeds[step] = v
    
    gs[step] = gravitational_acceleration(state[2])
    wind_vec = wind_velocity(state[2])
    winds[step] = np.linalg.norm(wind_vec)
    densities[step] = air_density(state[2])
    
    stage_idx, _ = get_current_stage(t)
    stage_info[step] = stage_idx
    
    if next_separation_idx < len(stages) - 1:
        separation_time = cumulative_times[next_separation_idx + 1]
        if separation_time - dt/2 <= t < separation_time + dt/2:
            if state[6] > stage_masses_after[next_separation_idx]:
                state[6] = stage_masses_after[next_separation_idx]
                print(f"  STAGE {next_separation_idx+1} SEPARATED at t={t:.2f}s, Height={state[2]/1000:.2f}km")
                print(f"    New mass: {state[6]:.0f} kg, Stage {next_separation_idx+2} activated")
            next_separation_idx += 1
        elif t >= separation_time + dt/2:
            next_separation_idx += 1
    
    if state[2] < 0:
        print(f"  Ground contact at t={t:.1f}s")
        actual_steps = step + 1
        break
    
    state = rk4_step(state, t, dt)
    t += dt
    
    if step % 1000 == 0:
        current_stage_idx, _ = get_current_stage(t)
        print(f"  t={t:.1f}s, h={state[2]/1000:.2f}km, v={v:.0f}m/s, m={state[6]:.0f}kg, Stage={current_stage_idx+1}")

print("\nSimulation complete!")

times = times[:actual_steps]
xs = xs[:actual_steps]
ys = ys[:actual_steps]
zs = zs[:actual_steps]
vxs = vxs[:actual_steps]
vys = vys[:actual_steps]
vzs = vzs[:actual_steps]
speeds = speeds[:actual_steps]
ms = ms[:actual_steps]
stage_info = stage_info[:actual_steps]
gs = gs[:actual_steps]
winds = winds[:actual_steps]
densities = densities[:actual_steps]

print("\n" + "="*80)
print("RESULTS")
print("="*80)

cum_time = 0.0
for i, stage in enumerate(stages):
    burn_time = stage['burn_time']
    if len(times) == 0:
        continue
    burn_end_idx = np.argmin(np.abs(times - (cum_time + burn_time)))
    
    if burn_end_idx < len(zs) and burn_end_idx < len(speeds) and burn_end_idx < len(ms):
        print(f"\n{stage['name']}:")
        print(f"  Burn duration: {burn_time:.1f}s ({cum_time:.1f}s - {cum_time + burn_time:.1f}s)")
        print(f"  At burn end:")
        print(f"    Height: {zs[burn_end_idx]/1000:.2f} km")
        print(f"    Velocity: {speeds[burn_end_idx]:.0f} m/s ({speeds[burn_end_idx]*3.6:.0f} km/h)")
        print(f"    Mass: {ms[burn_end_idx]:.0f} kg")
        print(f"    Gravity: {gs[burn_end_idx]:.3f} m/s^2")
        print(f"    Wind: {winds[burn_end_idx]:.1f} m/s")
    cum_time += burn_time

if len(zs) == 0:
    raise RuntimeError("Simulation produced no data points")
apogee_idx = np.argmax(zs)
print(f"\nApogee:")
print(f"  Max height: {zs[apogee_idx]/1000:.2f} km")
print(f"  Time: {times[apogee_idx]:.1f} s")
if apogee_idx < len(speeds):
    print(f"  Velocity: {speeds[apogee_idx]:.0f} m/s")
if apogee_idx < len(xs) and apogee_idx < len(ys):
    print(f"  Horizontal range: {np.sqrt(xs[apogee_idx]**2 + ys[apogee_idx]**2)/1000:.2f} km")
print("="*80)

fig = plt.figure(figsize=(20, 14))

ax1 = fig.add_subplot(3, 4, 1, projection='3d')
ax1.plot(xs/1000, ys/1000, zs/1000, 'b-', linewidth=2, label='Trajectory')
ax1.scatter(xs[0]/1000, ys[0]/1000, zs[0]/1000, color='green', s=100, marker='o', label='Start')
ax1.set_xlabel('X (km)')
ax1.set_ylabel('Y (km)')
ax1.set_zlabel('Z Height (km)')
ax1.set_title('3D Trajectory')
ax1.legend()

ax2 = fig.add_subplot(3, 4, 2)
ax2.plot(times, zs/1000, 'b-', linewidth=2, label='Height')
cum_time = 0.0
for i, stage in enumerate(stages):
    burn_time = stage['burn_time']
    ax2.axvline(cum_time + burn_time, color='r', linestyle='--', linewidth=2,
                label=f"Stage {i+1} End" if i == 0 else '')
    burn_end_idx = np.argmin(np.abs(times - (cum_time + burn_time)))
    if burn_end_idx < len(zs):
        ax2.plot(cum_time + burn_time, zs[burn_end_idx]/1000, 'ro', markersize=8)
    cum_time += burn_time
ax2.plot(times[apogee_idx], zs[apogee_idx]/1000, 'y*', markersize=15, label='Apogee')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Height (km)')
ax2.set_title('Height vs Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(3, 4, 3)
ax3.plot(times, speeds, 'r-', linewidth=2, label='Velocity')
cum_time = 0.0
for i, stage in enumerate(stages):
    burn_time = stage['burn_time']
    ax3.axvline(cum_time + burn_time, color='g', linestyle='--', linewidth=1.5)
    burn_end_idx = np.argmin(np.abs(times - (cum_time + burn_time)))
    if burn_end_idx < len(speeds):
        ax3.plot(cum_time + burn_time, speeds[burn_end_idx], 'go', markersize=6)
    cum_time += burn_time
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Velocity (m/s)')
ax3.set_title('Velocity')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(3, 4, 4)
ax4.plot(times, ms, 'm-', linewidth=2, label='Mass')
cum_time = 0.0
for i, stage in enumerate(stages):
    cum_time += stage['burn_time']
    ax4.axvline(cum_time, color='r', linestyle='--', linewidth=1.5)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Mass (kg)')
ax4.set_title('Mass with Stage Separation')
ax4.set_xlim([0, min(100, times[-1])])
ax4.grid(True, alpha=0.3)
ax4.legend()

ax5 = fig.add_subplot(3, 4, 5)
ax5.plot(zs/1000, gs, 'g-', linewidth=2)
ax5.axhline(g0, color='k', linestyle=':', alpha=0.5, label=f'g0 = {g0:.2f} m/s^2')
ax5.set_xlabel('Height (km)')
ax5.set_ylabel('Gravity (m/s^2)')
ax5.set_title('Height-Dependent Gravity')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(3, 4, 6)
ax6.plot(zs/1000, winds, 'c-', linewidth=2)
ax6.set_xlabel('Height (km)')
ax6.set_ylabel('Wind Speed (m/s)')
ax6.set_title('Wind Profile (Jetstream at ~10km)')
ax6.grid(True, alpha=0.3)

ax7 = fig.add_subplot(3, 4, 7)
ax7.semilogy(zs/1000, densities, 'orange', linewidth=2)
ax7.set_xlabel('Height (km)')
ax7.set_ylabel('Air Density (kg/m^3)')
ax7.set_title('ISA Atmosphere - Air Density')
ax7.grid(True, alpha=0.3)

ax8 = fig.add_subplot(3, 4, 8)
for i, stage in enumerate(stages):
    start = cumulative_times[i]
    end = cumulative_times[i+1]
    ax8.barh(0, end - start, left=start, label=stage['name'], alpha=0.7)
ax8.set_xlabel('Time (s)')
ax8.set_title('Stage Timeline')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='x')

ax9 = fig.add_subplot(3, 4, 9)
ax9.plot(times, stage_info + 1, 'c-', linewidth=2)
ax9.set_xlabel('Time (s)')
ax9.set_ylabel('Active Stage')
ax9.set_title('Stage Activation')
ax9.set_ylim([0.5, len(stages) + 0.5])
ax9.grid(True, alpha=0.3)

ax10 = fig.add_subplot(3, 4, 10)
ax10.plot(xs/1000, zs/1000, 'g-', linewidth=2)
ax10.set_xlabel('X (km)')
ax10.set_ylabel('Z Height (km)')
ax10.set_title('Trajectory (X-Z Plane)')
ax10.grid(True, alpha=0.3)

ax11 = fig.add_subplot(3, 4, 11)
ax11.plot(times, vxs, 'r-', linewidth=1.5, label='vx')
ax11.plot(times, vys, 'b-', linewidth=1.5, label='vy')
ax11.plot(times, vzs, 'g-', linewidth=1.5, label='vz')
ax11.set_xlabel('Time (s)')
ax11.set_ylabel('Velocity (m/s)')
ax11.set_title('Velocity Components')
ax11.legend()
ax11.grid(True, alpha=0.3)

ax12 = fig.add_subplot(3, 4, 12)
potential_energy = ms * g0 * zs / 1e9
kinetic_energy = 0.5 * ms * speeds**2 / 1e9
total_energy = potential_energy + kinetic_energy
ax12.plot(times, potential_energy, 'g-', linewidth=1.5, label='Potential')
ax12.plot(times, kinetic_energy, 'r-', linewidth=1.5, label='Kinetic')
ax12.plot(times, total_energy, 'b-', linewidth=2, label='Total')
ax12.set_xlabel('Time (s)')
ax12.set_ylabel('Energy (GJ)')
ax12.set_title('Energy vs Time')
ax12.legend()
ax12.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.96, bottom=0.05)
plt.savefig('rocket_staging_enhanced.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nEnhanced Staging Simulation complete!")
print(f"Plots saved as: rocket_staging_enhanced.png")
