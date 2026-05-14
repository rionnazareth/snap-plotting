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

# ── constants & paths ─────────────────────────────────────────────────────────
BASE_PATH  = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion'
SNAPBASE   = 'snap_'

RUNS = {
    r'Hydro only':      {'path': BASE_PATH + '/new/output_cnocr/',    'ls': '--',  'c': 'maroon'},
    r'Hydro+B fields':       {'path': BASE_PATH + '/new/output_cbf/',    'ls': ':',   'c': 'orange'},
    r'Hydro+B fields+CRs':      {'path': BASE_PATH + '/new/output_cbcr/',   'ls': '-',   'c': 'teal'},
}

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'beta')
os.makedirs(OUTDIR, exist_ok=True)

from lib import *

R_RANGE = (1e-3, 1)   # kpc
NBINS   = 500

def _count_snaps(path):
    n = 0
    while os.path.exists(os.path.join(path, f'{SNAPBASE}{n:03d}.hdf5')):
        n += 1
    return n

# ── load ALL snapshots for both runs ─────────────────────────────────────────
print('Loading all snapshots...')

sim_data = {}
for label, cfg in RUNS.items():
    sim_data[label] = {'snaps': [], 'times': []}
    n_snaps = _count_snaps(cfg['path'])
    for n in range(n_snaps):
        try:
            s = load_snap_data(n, snappath=cfg['path'], snapbase=SNAPBASE)
            sim_data[label]['snaps'].append(s)
            sim_data[label]['times'].append(calc_snap_time(s))
            print(f"  {label} snap {n:03d}  {sim_data[label]['times'][-1]:.1f} Myr")
        except Exception as e:
            print(f"  {label} snap {n:03d} missing. {e}")

print('\nPlotting density peak radius evolution...')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_r, ax_log = axes

def find_density_shell(snaps, times, r_range, nbins, tol=1e-3):
    rho_shell, t_arr = [], []
    shell_t_arr = []
    for n, (s, t) in enumerate(zip(snaps, times)):
        r, rho = radial_profile_lin(s, 'rho', r_range=r_range, nbins=nbins)
        
        if r is None or not np.any(np.isfinite(rho)):
            continue
            
        idx = np.nanargmax(rho)
        grad = np.abs(np.gradient(rho, r))
        grad_threshold = tol * np.nanmax(grad)
        
        # left = idx
        # while left > 0 and (grad[left] > grad_threshold or left >= idx - 2):
        #     if left < idx - 2 and grad[left] <= grad_threshold: break
        #     left -= 1
            
        # right = idx
        # while right < len(rho) - 1 and (grad[right] > grad_threshold or right <= idx + 2):
        #     if right > idx + 2 and grad[right] <= grad_threshold: break
        #     right += 1

        range = (r[idx],r[-1])
        r_shock, r_reverse_shock = find_shock_radius(s, r_range=range, nbins=nbins)
        r_shell_l, r_shell_u = find_shell_radius(s)

        mask = (r >= r_shell_u) & (r <= r_shock)
            
        avg_rho = np.nanmean(rho[mask])
        rho_shell.append(avg_rho)
        t_arr.append(t)
        shell_t_arr.append(r_shock - r_shell_u)
    return np.array(t_arr), np.array(rho_shell), np.array(shell_t_arr)

norm = True

# Process shells 
for label in RUNS:
    t_arr, rho_arr, shell_t_arr = find_density_shell(sim_data[label]['snaps'], sim_data[label]['times'], R_RANGE, NBINS)
    if len(rho_arr) > 0 and norm:
        rho0 = rho_arr[0]
        rho_arr /= rho0
    
    t = 0.0
    mask = t_arr > t
    sim_data[label]['t_arr'] = t_arr[mask]
    sim_data[label]['rho_arr'] = rho_arr[mask]
    sim_data[label]['shell_t_arr'] = shell_t_arr[mask]

for ax in axes:
    for label, cfg in RUNS.items():
        if 't_arr' in sim_data[label] and len(sim_data[label]['t_arr']) > 0:
            ax.plot(sim_data[label]['t_arr'], sim_data[label]['rho_arr'], ls=cfg['ls'], color=cfg['c'], lw=2, label=label)
    
    ax.set_xlabel('Time [Myr]', fontsize=11)
    if norm: ax.set_ylabel(r'$\rho_{sh}/\rho_0$', fontsize=11) 
    else: ax.set_ylabel(r'Density [code units]', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, ls='--')

ax_log.set_xscale('log')
ax_log.set_yscale('log')

axes[0].set_title('Density shell radius vs time (linear)')
ax_log.set_title('Density shell radius vs time (log–log)')

fig.suptitle('Evolution of density shell position', fontsize=13)
plt.tight_layout()
fname = os.path.join(OUTDIR, 'evolution_density_shell_radius.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')

print('\nPlotting shell thickness evolution...')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_lin, ax_log = axes

for ax in axes:
    for label, cfg in RUNS.items():
        if 't_arr' in sim_data[label] and len(sim_data[label]['t_arr']) > 0:
            ax.plot(sim_data[label]['t_arr'], sim_data[label]['shell_t_arr'], ls=cfg['ls'], color=cfg['c'], lw=2, label=label)
    
    ax.set_xlabel('Time [Myr]', fontsize=11)
    ax.set_ylabel('Shell thickness [kpc]', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, ls='--')

ax_log.set_xscale('log')
ax_log.set_yscale('log')

axes[0].set_title('Shell thickness vs time (linear)')
ax_log.set_title('Shell thickness vs time (log–log)')

fig.suptitle('Evolution of density shell thickness', fontsize=13)
plt.tight_layout()
fname = os.path.join(OUTDIR, 'evolution_shell_thickness.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')
