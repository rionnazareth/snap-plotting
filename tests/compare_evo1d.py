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

Saved to: test_ai/plots/
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
NBINS   = 300

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

# ── one combined figure per quantity ─────────────────────────────────────────
for field, label, log_y in QUANTITIES:

    # Check at least one snapshot has this field
    has_field = any(field in s.data for s in snaps_nocr + snaps_cr)
    if not has_field:
        print(f'  Skipping {field}: not present in any snapshot')
        continue

    print(f'\nPlotting evolution: {field}')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    ax_nocr, ax_cr, ax_diff = axes

    # colour maps: early → dark, late → bright
    cmap_nocr = cm.get_cmap('Greys_r', N_COMMON)
    cmap_cr   = cm.get_cmap('Blues_r', N_COMMON)

    # ---- left panel: no-CR evolution ----
    r_last_nocr, prof_last_nocr = None, None
    for n, s in enumerate(snaps_nocr[:N_COMMON]):
        r, prof = radial_profile_log(s, field, r_range=R_RANGE, nbins=NBINS)
        if r is None:
            continue
        if log_y:
            prof = np.where(prof <= 0, np.nan, prof)
        color = cmap_nocr(n / max(N_COMMON - 1, 1))
        lw    = 1.5 + 0.5 * (n / max(N_COMMON - 1, 1))
        ax_nocr.plot(r, prof, color=color, lw=lw, alpha=0.9,
                   label=f'snap {n:02d}  ({times_nocr[n]:.0f} Myr)')
        if n == N_COMMON - 1:
            r_last_nocr, prof_last_nocr = r, prof

    ax_nocr.set_title('output_nocr (Hydro only)', fontsize=11)
    ax_nocr.set_xlabel('Radius [kpc]', fontsize=10)
    ax_nocr.set_ylabel(label, fontsize=10)
    ax_nocr.set_xscale('log')
    if log_y:
        ax_nocr.set_yscale('log')
    ax_nocr.legend(fontsize=7, loc='best', framealpha=0.6)
    ax_nocr.grid(True, which='both', alpha=0.25, ls='--')

    # ---- middle panel: CR evolution ----
    r_last_cr, prof_last_cr = None, None
    for n, s in enumerate(snaps_cr[:N_COMMON]):
        r, prof = radial_profile_log(s, field, r_range=R_RANGE, nbins=NBINS)
        if r is None:
            continue
        if log_y:
            prof = np.where(prof <= 0, np.nan, prof)
        color = cmap_cr(n / max(N_COMMON - 1, 1))
        lw    = 1.5 + 0.5 * (n / max(N_COMMON - 1, 1))
        ax_cr.plot(r, prof, color=color, lw=lw, alpha=0.9,
                     label=f'snap {n:02d}  ({times_cr[n]:.0f} Myr)')
        if n == N_COMMON - 1:
            r_last_cr, prof_last_cr = r, prof

    ax_cr.set_title('output_cr (CRs included)', fontsize=11)
    ax_cr.set_xlabel('Radius [kpc]', fontsize=10)
    ax_cr.set_xscale('log')
    if log_y:
        ax_cr.set_yscale('log')
    ax_cr.legend(fontsize=7, loc='best', framealpha=0.6)
    ax_cr.grid(True, which='both', alpha=0.25, ls='--')

    # ---- right panel: relative difference at last common snapshot ----
    if r_last_nocr is not None and r_last_cr is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = (prof_last_cr - prof_last_nocr) / np.abs(prof_last_nocr)

        ax_diff.plot(r_last_nocr, rel, color='purple', lw=2)
        ax_diff.axhline(0, color='k', lw=0.8, ls='--')
        ax_diff.set_title(
            f'Relative diff at snap {N_COMMON-1:02d}\n'
            r'$(CR - noCR)\,/\,|noCR|$', fontsize=10)
        ax_diff.set_xlabel('Radius [kpc]', fontsize=10)
        ax_diff.set_ylabel('Relative difference', fontsize=10)
        ax_diff.set_xscale('log')
        ax_diff.grid(True, which='both', alpha=0.25, ls='--')

    else:
        ax_diff.text(0.5, 0.5, 'Data unavailable',
                     ha='center', va='center', transform=ax_diff.transAxes)
        ax_diff.set_axis_off()

    fig.suptitle(f'Evolution comparison — {label}', fontsize=13)
    # plt.tight_layout()

    fname = os.path.join(OUTDIR, f'evolution_{field}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {fname}')

print('\nDone.')
