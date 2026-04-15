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

PATH_NOCR = BASE_PATH + '/old/output_homo/'
PATH_CR   = BASE_PATH + '/old/output_cr600/'
N_SNAPS   = 11

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clean')
os.makedirs(OUTDIR, exist_ok=True)

from lib import *

# ── quantity catalogue ────────────────────────────────────────────────────────
# (field_key,   label,                   log_y?)
QUANTITIES = [
    ('rho',       r'Density $\rho$ [code]',      True),
    ('pres',      r'Thermal pressure [code]',     True),
    ('crpres',    r'CR pressure [code]',          True),
    ('cren',      r'CR energy density [code]',    True),
    ('xcr',       r'CR pressure fraction',        True),
    ('u',         r'Internal energy [code]',      True),
    ('speed',     r'Speed |v| [code km/s]',       True),
    ('mach',      r'Mach number',                 True),
    ('vrad',      r'Radial velocity [code km/s]', False),
]

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

t_nocr_arr, rho_nocr_arr, r_nocr_arr = find_density_peak(snaps_nocr, times_nocr, R_RANGE, NBINS)
t_cr_arr,   rho_cr_arr,   r_cr_arr   = find_density_peak(snaps_cr,   times_cr,   R_RANGE, NBINS)

t = 0.0
mask = (t_nocr_arr > t) & (t_cr_arr > t)  # exclude early snapshots 
t_nocr_arr = t_nocr_arr[mask]
r_nocr_arr = r_nocr_arr[mask]
rho_nocr_arr = rho_nocr_arr[mask]
t_cr_arr = t_cr_arr[mask]
r_cr_arr = r_cr_arr[mask]
rho_cr_arr = rho_cr_arr[mask]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
ax_lin, ax_log = axes

s_nocr = 5 + 100 * (r_nocr_arr / np.nanmax(r_nocr_arr))
s_cr = 5 + 100 * (r_cr_arr / np.nanmax(r_cr_arr))

for ax in axes:
    sc1 = ax.scatter(t_nocr_arr, rho_nocr_arr, c=r_nocr_arr, cmap='Blues', s=s_nocr, 
                     marker='o', zorder=3)
    sc2 = ax.scatter(t_cr_arr, rho_cr_arr, c=r_cr_arr, cmap='Blues', s=s_cr, 
                     marker='o', zorder=3)
    
    ax.plot(t_nocr_arr, rho_nocr_arr, ':', color='maroon', lw=1, alpha=0.5, label='no CRs (Hydro only)')
    ax.plot(t_cr_arr, rho_cr_arr, '-', color='steelblue', lw=1, alpha=0.5, label='CRs included')

    ax.set_xlabel('Time [Myr]', fontsize=11)
    ax.set_ylabel('Peak density [code units]', fontsize=11)
    ax.grid(True, alpha=0.3, ls='--')

ax_log.set_xscale('log')
ax_log.set_yscale('log')

axes[0].legend(fontsize=9)
axes[0].set_title('Peak Density vs Time (linear)')
ax_log.set_title('Peak Density vs Time (log–log)')

cbar1 = fig.colorbar(sc1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
cbar1.set_label('Peak radius [kpc]')
cbar2 = fig.colorbar(sc2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
cbar2.set_label('Peak radius [kpc]')

fig.suptitle('Evolution of Peak Density vs Time', fontsize=13)
fname = os.path.join(OUTDIR, 'evolution_peak_density_vs_time.png')
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')
t_nocr_arr, rho_nocr_arr, r_nocr_arr = find_density_peak(snaps_nocr, times_nocr, R_RANGE, NBINS)
t_cr_arr,   rho_cr_arr,   r_cr_arr   = find_density_peak(snaps_cr,   times_cr,   R_RANGE, NBINS)

t = 0.0
mask = (t_nocr_arr > t) & (t_cr_arr > t)  # exclude early snapshots 
t_nocr_arr = t_nocr_arr[mask]
r_nocr_arr = r_nocr_arr[mask]
rho_nocr_arr = rho_nocr_arr[mask]
t_cr_arr = t_cr_arr[mask]
r_cr_arr = r_cr_arr[mask]
rho_cr_arr = rho_cr_arr[mask]
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
ax_lin, ax_log = axes

s_nocr = 5 + 100 * (r_nocr_arr / np.nanmax(r_nocr_arr))
s_cr = 5 + 100 * (r_cr_arr / np.nanmax(r_cr_arr))

for ax in axes:
    sc1 = ax.scatter(r_nocr_arr, rho_nocr_arr, c=t_nocr_arr, cmap='Blues', s=s_nocr, 
                     marker='^', label='no CRs (Hydro only)', zorder=3)
    sc2 = ax.scatter(r_cr_arr, rho_cr_arr, c=t_cr_arr, cmap='Blues', s=s_cr, 
                     marker='o', label='CRs included', zorder=3)
    
    # Optional connecting lines to show the trajectory
    ax.plot(r_nocr_arr, rho_nocr_arr, ':', color='maroon', lw=1, alpha=0.5)
    ax.plot(r_cr_arr, rho_cr_arr, '-', color='steelblue', lw=1, alpha=0.5)

    ax.set_xlabel('Peak radius [kpc]', fontsize=11)
    ax.set_ylabel('Peak density [code units]', fontsize=11)
    ax.grid(True, alpha=0.3, ls='--')

ax_log.set_xscale('log')
ax_log.set_yscale('log')

axes[0].legend(fontsize=9)
axes[0].set_title('Density vs Radius (linear)')
ax_log.set_title('Density vs Radius (log–log)')

cbar = fig.colorbar(sc2, ax=axes, orientation='horizontal', fraction=0.05, pad=0.15)
cbar.set_label('Time [Myr]')

fig.suptitle('Evolution of Peak Density vs Radius', fontsize=13)
fname = os.path.join(OUTDIR, 'evolution_density_vs_radius.png')
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')
