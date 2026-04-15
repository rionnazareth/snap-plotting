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

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crevo')
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

# Track position of Mach-number peak across snapshots for both runs
print('\nPlotting shock radius evolution...')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_r, ax_log = axes

def find_shock_radius(snaps, times, r_range, nbins):
    r_shock, t_arr = [], []
    for n, (s, t) in enumerate(zip(snaps, times)):
        r, mach = radial_profile_lin(s, 'mach', r_range=r_range, nbins=nbins)
        if r is None or not np.any(np.isfinite(mach)):
            continue
        idx = np.nanargmax(mach)
        r_shock.append(r[idx])
        t_arr.append(t)
    return np.array(t_arr), np.array(r_shock)

t_nocr_arr, r_nocr_arr = find_shock_radius(snaps_nocr, times_nocr, R_RANGE, NBINS)
t_cr_arr,   r_cr_arr   = find_shock_radius(snaps_cr,   times_cr,   R_RANGE, NBINS)

t = 0.0
mask = (t_nocr_arr > t) & (t_cr_arr > t)  # exclude early snapshots 
t_nocr_arr = t_nocr_arr[mask]
r_nocr_arr = r_nocr_arr[mask]
t_cr_arr = t_cr_arr[mask]
r_cr_arr = r_cr_arr[mask]

for ax in axes:
    ax.plot(t_nocr_arr, r_nocr_arr, '^:', color='maroon',
            lw=2, label='no CRs (Hydro only)')
    ax.plot(t_cr_arr,   r_cr_arr,   'o-', color='steelblue',
            lw=2, label='CRs included')
    
    ax.set_xlabel('Time [Myr]', fontsize=11)
    ax.set_ylabel('Shock radius [kpc]', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, ls='--')

# Add power-law reference lines to the log-log panel
ax_log.set_xscale('log')
ax_log.set_yscale('log')
if len(t_nocr_arr) > 1 and np.any(t_nocr_arr > 0):
    idx0   = np.argmax(t_nocr_arr > 0)   # first snapshot with t > 0
    t0, r0 = t_nocr_arr[idx0], r_nocr_arr[idx0]
    t_max  = max([t[-1] for t in (t_cr_arr, t_nocr_arr) if len(t) > 0])
    t_ref  = np.linspace(t0, t_max, 200)
    for ax in axes:
        ax.plot(t_ref, r0 * (t_ref/t0)**1,     'k--', lw=1, alpha=0.6,
                label=r'$r \propto t$')
        ax.plot(t_ref, r0 * (t_ref/t0)**(3/5), 'k:',  lw=1, alpha=0.6,
                label=r'$r \propto t^{3/5}$')
        ax.legend(fontsize=9)

axes[0].set_title('Shock radius vs time (linear)')
ax_log.set_title('Shock radius vs time (log–log)')

fig.suptitle('Evolution of shock front position', fontsize=13)
plt.tight_layout()
fname = os.path.join(OUTDIR, 'evolution_shock_radius.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')

print('\nDone.')
