import matplotlib.pyplot as plt
import scienceplots
from lib import load_snap_data, calc_snap_time, find_Rcool, find_shell_radius

plt.style.use(['science'])

BASE_PATH = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion'

# Single dictionary: { 'Label': ('Path_to_output', 'color') }
simulations = {
    # 'diff+uni B': (BASE_PATH + '/output_cdiff/', 'tab:blue'),
    # 'diff only': (BASE_PATH + '/output_cnob/', 'tab:orange'),
    'no diffusion': (BASE_PATH + '/output_cool/', 'tab:green'),
    # 'diff+turb B': (BASE_PATH + '/output_cturb/', 'tab:red')
}

snaps = range(1, 11)

plt.figure(figsize=(6, 4))

for label, (path, color) in simulations.items():
    times = []
    r_cools = []
    r_shell = []
    rdot = []
    
    print(f"Processing {label}...")
    for snap in snaps:
    # try:
        # We only use load_snap_data here to get the physical time of the snapshot
        s = load_snap_data(snap, snappath=path, snapbase='snap_')
        t = calc_snap_time(s)
        r_shell_l, r_shell_u = find_shell_radius(s)
        
        # find_Rcool calculates the cooling radius using lib.py's internal routine
        r_cff_kpc, Rdotsh_kms, n0 = find_Rcool(snappath=path, snapnum=snap)
        
        times.append(t)
        r_cools.append(r_cff_kpc)
        r_shell.append(r_shell_l)  # Using the lower bound of the shell radius as a reference
        rdot.append(Rdotsh_kms)

        # except Exception as e:
        #     print(f"Could not process snap {snap} for {label}: {e}")
            
    # Plot the evolution for this simulation with explicit color
    plt.loglog(times, r_cools, marker='o', label=f'Cooling Radius', color=color)
    plt.loglog(times, r_shell, marker='x', linestyle='--', label=f'Shell Radius', color='maroon')
    # plt.loglog(times, rdot, marker='s', linestyle='-.', label=f'Shock Speed', color='purple')

plt.xlabel('Time [Myr]')
plt.ylabel(r'$r_\mathrm{cool}$ [kpc]')
plt.title('Time Evolution of Cooling Radius')
plt.legend()
plt.tight_layout()

output_file = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/snap-plotting/tests/rad/rcool_evolution.png'
plt.savefig(output_file, dpi=200)
print(f"Saved plot to {output_file}")