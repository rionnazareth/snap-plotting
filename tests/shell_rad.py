"""
compare_1d_evolution.py
=======================
Shows how each simulation evolves over time and contrasts the two runs:

  - output_cr     (no CR diffusion)
  - output_crexps (with CR diffusion)

Inspired by the shocks1d.ipynb notebook: for every snapshot we compute a
logarithmically-binned radial profile and overplot all snapshots on one axis,
coloured from early (dark) to late (bright).  Then we draw the same plot for
the other run beside it so the differences are immediately visible.

Figures produced
----------------
  evolution_<quantity>.png     – N rows of 3 panels each:
                                   [0] no-diffusion  evolution
                                   [1] with-diffusion evolution
                                   [2] relative difference at the last
                                       common snapshot:  (crexps − cr)/cr

Quantities: rho, pres, crpres (from cren), u (internal energy), speed, mach

"""

import sys, os
sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/arepo-snap-util')
sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/gasCloudNfw/snap-plotting')

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from lib import *

plt.style.use(['science'])

# ── paths & settings ─────────────────────────────────────────────────────────
BASE_PATH   = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/old'
RUNS = {
    'Hydro (output_homo)':    {'path': f'{BASE_PATH}/output_homo',  'c': 'maroon',     'ls': ':'},
    'CRs (output_cr600)':     {'path': f'{BASE_PATH}/output_cr600', 'c': 'forestgreen','ls': '--'},
    'CR diffusion (output_cr)': {'path': f'{BASE_PATH}/output_cr',    'c': 'steelblue',  'ls': '-'},
}
SNAPBASE    = 'snap_'
N_SNAPS     = 11
OUTDIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fii')
os.makedirs(OUTDIR, exist_ok=True)

# ── extract shell radii ───────────────────────────────────────────────────────
print('Loading snapshots and extracting shell radii...')
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax_lin, ax_log = axes

for label, info in RUNS.items():
    t_arr, r_lower, r_upper = [], [], []
    
    for n in range(N_SNAPS):
        try:
            # load_snap_data is available from lib.py natively
            s = load_snap_data(n, snappath=info['path'] + '/', snapbase=SNAPBASE)
            # find_shell_radius uses wind tracer > 0.5 and calculates the percentiles.
            r_l, r_u = find_shell_radius(s)
            
            if not np.isnan(r_l) and not np.isnan(r_u):
                t_arr.append(calc_snap_time(s))
                r_lower.append(r_l)
                r_upper.append(r_u)
                
        except Exception as e:
            print(f'  {label} snap {n:03d} missing or error.')

    if len(t_arr) == 0:
        continue
        
    t_arr = np.array(t_arr)
    r_lower, r_upper = np.array(r_lower), np.array(r_upper)

    # Clean up t=0 out of arrays for stable plotting
    mask = t_arr > 0
    t_arr, r_lower, r_upper = t_arr[mask], r_lower[mask], r_upper[mask]

    # Plot thick line (filled between the 95th and 97th/99.7th percentiles)
    for ax in axes:
        # Plot central reference line
        ax.plot(t_arr, (r_lower + r_upper)/2, color=info['c'], ls=info['ls'], label=label)
        # Represents the shell thickness
        ax.fill_between(t_arr, r_lower, r_upper, color=info['c'], alpha=0.3, lw=0)

# ── formatting ─────────────────────────────────────────────────────────────
for ax in axes:
    ax.set_xlabel('Time [Myr]', fontsize=11)
    ax.set_ylabel('Shell radius [kpc]', fontsize=11)
    ax.grid(True, alpha=0.3, ls='--')

ax_lin.legend(fontsize=9, loc='upper left')
ax_log.set_xscale('log')
ax_log.set_yscale('log')

ax_lin.set_title('Shell radius vs time (linear)')
ax_log.set_title('Shell radius vs time (log–log)')
fig.suptitle('Evolution of shell front position (95th - 97th percentile)', fontsize=13)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'evolution_shell_radius.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> Saved output {fname}')
