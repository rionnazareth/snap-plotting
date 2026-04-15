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
GAMMA      = 5./3
GAMMA_CR   = 4./3
k_B        = 1.381e-16
m_p        = 1.66e-24
HYDROGENMASS_FRAC = 0.76

PATH_NOCR = BASE_PATH + '/old/output_cr600/'
PATH_CR   = BASE_PATH + '/output_cool/'
N_SNAPS   = 11

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clean')
os.makedirs(OUTDIR, exist_ok=True)

from lib import *


R_RANGE = (1e-3, 1)   # kpc
NBINS   = 500

# ── load ALL snapshots for both runs ─────────────────────────────────────────
print('Loading all snapshots...')

snaps_nocr, snaps_cr = [], []
times_nocr, times_cr = [], []

for n in range(N_SNAPS):
    try:
        s = load_snap_data(n, snappath=PATH_NOCR, snapbase=SNAPBASE)
        snaps_nocr.append(s)
        times_nocr.append(calc_snap_time(s))
        print(f'  output_nocr snap {n:03d}  {times_nocr[-1]:.1f} Myr')
    except Exception as e:
        print(f'  output_nocr snap {n:03d} missing.')

for n in range(len(snaps_nocr)):
    try:
        s = load_snap_data(n, snappath=PATH_CR, snapbase=SNAPBASE)
        snaps_cr.append(s)
        times_cr.append(calc_snap_time(s))
        print(f'  output_cr   snap {n:03d}  {times_cr[-1]:.1f} Myr')
    except Exception as e:
        print(f'  output_cr snap {n:03d} missing.')

N_COMMON = min(len(snaps_nocr), len(snaps_cr), N_SNAPS)

# Track position of density peak across snapshots for both runs
print('\nPlotting density peak radius evolution...')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_r, ax_log = axes

def find_density_peak(snaps, times, r_range, nbins):
    rho_peak, r_peak, t_arr = [], [], []
    for n, (s, t) in enumerate(zip(snaps, times)):
        r, rho = radial_profile_lin(s, 'rho', r_range=r_range, nbins=nbins)
        if r is None or not np.any(np.isfinite(rho)):
            continue
        idx = np.nanargmax(rho)
        rho_peak.append(rho[idx])
        r_peak.append(r[idx])
        t_arr.append(t)
    return np.array(t_arr), np.array(rho_peak), np.array(r_peak)

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
        
        # Expand left from the peak
        left = idx
        while left > 0 and (grad[left] > grad_threshold or left >= idx - 2):
            if left < idx - 2 and grad[left] <= grad_threshold: break
            left -= 1
            
        # Expand right from the peak
        right = idx
        while right < len(rho) - 1 and (grad[right] > grad_threshold or right <= idx + 2):
            if right > idx + 2 and grad[right] <= grad_threshold: break
            right += 1
            
        avg_rho = np.nanmean(rho[left:right+1])
        rho_shell.append(avg_rho)
        t_arr.append(t)
        shell_t_arr.append(r[right] - r[left])
    return np.array(t_arr), np.array(rho_shell), np.array(shell_t_arr)

t_nocr_arr, rho_nocr_arr, shell_t_nocr_arr = find_density_shell(snaps_nocr, times_nocr, R_RANGE, NBINS)
t_cr_arr,   rho_cr_arr,   shell_t_cr_arr   = find_density_shell(snaps_cr,   times_cr,   R_RANGE, NBINS)

# divide by the initial density to get dimensionless values
norm = True
if len(rho_nocr_arr) > 0 and norm:
    rho0 = rho_nocr_arr[0]
    rho_nocr_arr /= rho0
    rho_cr_arr   /= rho0
t = 0.0
mask = (t_nocr_arr > t) & (t_cr_arr > t)  # exclude early snapshots 
t_nocr_arr = t_nocr_arr[mask]
rho_nocr_arr = rho_nocr_arr[mask]
shell_t_nocr_arr = shell_t_nocr_arr[mask]
t_cr_arr = t_cr_arr[mask]
rho_cr_arr = rho_cr_arr[mask]
shell_t_cr_arr = shell_t_cr_arr[mask]

for ax in axes:
    ax.plot(t_nocr_arr, rho_nocr_arr, '^:', color='maroon',
            lw=2, label='no CRs (Hydro only)')
    ax.plot(t_cr_arr,   rho_cr_arr,   'o-', color='steelblue',
            lw=2, label='CRs included')
    
    ax.set_xlabel('Time [Myr]', fontsize=11)
    if norm: ax.set_ylabel(r'$\rho_{sh}/\rho_0$', fontsize=11) 
    else: ax.set_ylabel(r'Density [code units]', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, ls='--')

# Add power-law reference lines to the log-log panel
ax_log.set_xscale('log')
ax_log.set_yscale('log')
# if len(t_nocr_arr) > 1 and np.any(t_nocr_arr > 0):
#     idx0   = np.argmax(t_nocr_arr > 0)   # first snapshot with t > 0
#     t0, r0 = t_nocr_arr[idx0], r_nocr_arr[idx0]
#     t_max  = max([t[-1] for t in (t_cr_arr, t_nocr_arr) if len(t) > 0])
#     t_ref  = np.linspace(t0, t_max, 200)
#     for ax in axes:
#         ax.plot(t_ref, r0 * (t_ref/t0)**1,     'k--', lw=1, alpha=0.6,
#                 label=r'$r \propto t$')
#         ax.plot(t_ref, r0 * (t_ref/t0)**(3/5), 'k:',  lw=1, alpha=0.6,
#                 label=r'$r \propto t^{3/5}$')
#         ax.legend(fontsize=9)

axes[0].set_title('Density shell radius vs time (linear)')
ax_log.set_title('Density shell radius vs time (log–log)')

fig.suptitle('Evolution of density shell position', fontsize=13)
plt.tight_layout()
fname = os.path.join(OUTDIR, 'evolution_density_shell_radius.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')

print('\nDone.')

# Track shell thickness across snapshots for both runs
print('\nPlotting shell thickness evolution...')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_lin, ax_log = axes

for ax in axes:
    ax.plot(t_nocr_arr, shell_t_nocr_arr, '^:', color='maroon',
            lw=2, label='no CRs (Hydro only)')
    ax.plot(t_cr_arr,   shell_t_cr_arr,   'o-', color='steelblue',
            lw=2, label='CRs included')
    
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