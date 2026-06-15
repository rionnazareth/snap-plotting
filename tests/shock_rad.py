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
import matplotlib.cm as cm
import astropy.units as u
import arepo_run as arun
import scienceplots
plt.style.use(['science'])

from lib import *

# ── paths & settings ─────────────────────────────────────────────────────────
BASE= '/cosma8/data/dp317/dc-naza3/homogeneous'
SNAPBASE  = 'snap_'
N_SNAPS   = 9

RUNS = {
    #     r'Hydro only':      {'path': BASE + '/new/output_cnocr/',    'has_cr': False,  'ls': '--','m': '^',  'c': 'maroon'},
    #   r'Hydro+B fields':       {'path': BASE + '/new/output_cbf/',    'has_cr': True,  'ls': ':', 'm': 's',  'c': 'orange'},
    # r'Hydro+B fields+CRs':      {'path': BASE + '/new/output_cbcr/',   'has_cr': True,  'ls': '-', 'm': 'o',  'c': 'teal'},
        # r'$\rho = \rho_0 \times 10$ et': {'path': BASE + '/et_backup/50/', 'has_cr': True, 'ls': '-', 'c': 'C0','m': 'o'},
        #         r'$\rho = \rho_0/10$': {'path': BASE + '/et_backup/0.5/', 'has_cr': True, 'ls': ':', 'c': 'C1','m': 'D'},
        # r'$\rho = \rho_0$': {'path': BASE + '/et_backup/5/', 'has_cr': True, 'ls': ':', 'c': 'C2','m': 'D'},
            r'hydro run': {'path': BASE + '/mtests/output_bf/', 'ls': '-', 'c': 'C3','m': 'o'},
    r'with CRs':    {'path': BASE + '/mtests/output_bfcr/', 'ls': '--', 'c': 'C4','m': 's'},
    
}

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mtests')
os.makedirs(OUTDIR, exist_ok=True)

# ── extract shock radii ───────────────────────────────────────────────────────
print('Loading snapshots and determining shock radii...')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_lin, ax_log = axes

first_run_t = None
first_run_r = None

for label, info in RUNS.items():
    t_arr, r_shock_arr = [], []
    
    for n in range(N_SNAPS):
        try:
            s = load_snap_data(n, snappath=info['path'], snapbase=SNAPBASE)
            # lib.py's find_shock_radius returns (forward_shock, reverse_shock)
            r_f, r_r = find_shock_radius(s, r_range=(1e-3, 1), nbins=500)
            
            if not np.isnan(r_f):
                t_arr.append(calc_snap_time(s))
                if label == 'CR diffusion (output_cr)':
                    r_shock_arr.append(r_r)
                else:
                    r_shock_arr.append(r_f)
        except Exception as e:
            print(f'  {label} snap {n:03d} missing or error.')
            
    if len(t_arr) == 0:
        continue
        
    t_arr = np.array(t_arr)
    r_shock_arr = np.array(r_shock_arr)
    
    # Exclude early snapshots (t=0) to prevent log-scaling issues
    mask = t_arr > 0
    t_arr, r_shock_arr = t_arr[mask], r_shock_arr[mask]
    
    # Save the first plotted run for calculating the reference power-laws later
    if first_run_t is None:
        first_run_t, first_run_r = t_arr, r_shock_arr
        
    for ax in axes:
        ax.plot(t_arr, r_shock_arr, marker=info['m'], color=info['c'],
                lw=2, ls=info['ls'], label=label)

# ── formatting ─────────────────────────────────────────────────────────────
for ax in axes:
    ax.set_xlabel('Time [Myr]', fontsize=11)
    ax.set_ylabel('Shock radius [kpc]', fontsize=11)
    ax.grid(True, alpha=0.3, ls='--')
    
# Add power-law reference lines to the log-log panel
ax_log.set_xscale('log')
ax_log.set_yscale('log')

if first_run_t is not None and len(first_run_t) > 0:
    t0, r0 = first_run_t[0], first_run_r[0]
    t_max  = np.max(first_run_t)
    t_ref  = np.linspace(t0, t_max, 200)
    
    for ax in axes:
        ax.plot(t_ref, r0 * (t_ref/t0)**1,     'k--', lw=1, alpha=0.6, label=r'$r \propto t$')
        ax.plot(t_ref, r0 * (t_ref/t0)**(3/5), 'k:',  lw=1, alpha=0.6, label=r'$r \propto t^{3/5}$')
        
ax_lin.legend(fontsize=9)
ax_log.legend(fontsize=9)

ax_lin.set_title('Shock radius vs time (linear)')
ax_log.set_title('Shock radius vs time (log–log)')

fig.suptitle('Evolution of forward shock front position', fontsize=13)
plt.tight_layout()
fname = os.path.join(OUTDIR, 'evolution_shock_radius.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> Saved output {fname}')
