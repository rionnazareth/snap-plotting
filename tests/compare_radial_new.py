"""
compare_radial.py
=================
1D radial profile comparison of multiple simulations.

Mirrors the style of radial_plots.py (linear radial bins, publication-quality
formatting) but overlays simulations on the same axis and adds a ratio
panel so the effect is quantified at every radius.

For every shared snapshot this script produces one PNG with:
  - Left column  : actual profiles
  - Right column : ratio (vs first simulation)

A final summary PNG overlays all snapshots per quantity on a 3×2 grid.
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import astropy.units as u
import arepo_run as arun

# ── constants & paths ─────────────────────────────────────────────────────────
BASE     = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion'
SNAPBASE = 'snap_'
GAMMA      = 5./3
GAMMA_CR   = 4./3
k_B        = 1.381e-16
m_p        = 1.66e-24
unit_v     = 1.651077e6#1.e5

RUNS = {
    'no CRs no B':   {'path': BASE + '/old/output_homo/',   'has_cr': False, 'ls': '-',  'c': 'lightcoral'},
    'only CRs no B': {'path': BASE + '/output_cr/',     'has_cr': True,  'ls': '--', 'c': 'steelblue'},
    r'with cooling only CRs':      {'path': BASE + '/output_cool/',    'has_cr': True,  'ls': '--',  'c': 'lightgreen'},
    r'$10^{-6} \mathrm{G}$':      {'path': BASE + '/output_uni/',    'has_cr': True,  'ls': ':',  'c': 'dimgrey'},
    r'$\kappa = 3\times 10^{26} \mathrm{cm}^2/\mathrm{s}$':     {'path': BASE + '/output_test/',   'has_cr': True,  'ls': '-.', 'c': 'steelblue'},
    # r'$10^{-3} \mathrm{G}$':     {'path': BASE + '/output_hb/', 'has_cr': True,  'ls': '-',  'c': 'darkgreen'},
}

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
os.makedirs(OUTDIR, exist_ok=True)

R_RANGE = (1e-3, 1)
NBINS   = 500

# ── quantity catalogue ────────────────────────────────────────────────────────
# (field, y-label, log_y)
QUANTITIES = [
    ('rho',       r'Density $\rho$ [code]',      False),
    ('pres',      r'Thermal pressure [code]',    True),
    ('xcr',    r'CR pressure/thermal pressure [code]',         True),
    ('cren',      r'CR energy density [code]',   True),
    ('temp',         r'Temperature [K]',     True),
    ('speed',     r'Speed |v| [code km/s]',      True),
    ('mach',      r'Mach number',                True),
    ('vrad',      r'Radial velocity [code km/s]', False),
    # ('bflden',    r'B-field energy density [code]', True),
    # ('bflds',     r'B-field strength [code]',         True)
]

# ── helpers ───────────────────────────────────────────────────────────────────
def snap_time_myr(s):
    Lu = s.header['UnitLength_in_cm'] * u.cm
    Vu = s.header['UnitVelocity_in_cm_per_s'] * u.cm / u.s
    return (s.header['Time'] * Lu / Vu).to(u.Myr).value
from lib import *


def apply_style():
    plt.rcParams.update({
        'font.family'     : 'serif',
        'mathtext.fontset': 'cm',
        'font.size'       : 11,
        'axes.labelsize'  : 12,
        'axes.titlesize'  : 13,
        'legend.fontsize' : 9,
        'axes.grid'       : True,
        'grid.alpha'      : 0.3,
        'grid.linestyle'  : '--',
        'lines.linewidth' : 2.0,
        'xtick.direction' : 'in',
        'ytick.direction' : 'in',
    })

def _count_snaps(path):
    n = 0
    while os.path.exists(os.path.join(path, f'{SNAPBASE}{n:03d}.hdf5')):
        n += 1
    return n

# ── load snapshots & compute profiles ─────────────────────────────────────────
apply_style()
print('Loading snapshots and computing profiles ...')

sims = {}
max_snaps = 0
for label, cfg in RUNS.items():
    n_snaps = _count_snaps(cfg['path'])
    if n_snaps > max_snaps:
        max_snaps = n_snaps
    sims[label] = {
        'n_snaps': n_snaps,
        'profiles': {f[0]: [] for f in QUANTITIES},
        'times': []
    }
# max_snaps = 2
for snap_num in range(max_snaps):
    for label, cfg in RUNS.items():
        if snap_num >= sims[label]['n_snaps']:
            for f, *_ in QUANTITIES:
                sims[label]['profiles'][f].append((None, None))
            sims[label]['times'].append(None)
            continue
            
        try:
            sc = load_snap_data(snap_num, snappath=cfg['path'], snapbase=SNAPBASE)
            t_myr = snap_time_myr(sc)
            sims[label]['times'].append(t_myr)
            
            for field, *_ in QUANTITIES:
                sims[label]['profiles'][field].append(radial_profile_lin(sc, field, r_range=R_RANGE, nbins=NBINS))
            print(f'  {label} snap {snap_num:03d} — {t_myr:.1f} Myr')
        except Exception as e:
            print(f"Skipping {label} snap {snap_num:03d}: {e}")
            for f, *_ in QUANTITIES:
                sims[label]['profiles'][f].append((None, None))
            sims[label]['times'].append(None)

# ── per-snapshot figures ──────────────────────────────────────────────────────
labels_list = list(RUNS.keys())
ref_label = labels_list[0]

for snap_num in range(max_snaps):
    active = []
    for f, ylabel, log_y in QUANTITIES:
        has_any = any(sims[l]['profiles'][f][snap_num][0] is not None for l in labels_list)
        if has_any:
            active.append((f, ylabel, log_y))

    if not active:
        continue

    n_qty = len(active)
    fig, axes = plt.subplots(n_qty, 2, figsize=(13, 3.8 * n_qty), squeeze=False)
    
    t_ref = sims[ref_label]['times'][snap_num] if sims[ref_label]['times'][snap_num] is not None else -1

    fig.suptitle(f'Radial profiles — snap {snap_num:03d} (Ref Time: {t_ref:.1f} Myr)', fontsize=12, y=1.002)
    axes[0, 0].set_title('Profiles', fontsize=10)
    axes[0, 1].set_title(f'Ratio to {ref_label}', fontsize=10)

    for row, (field, ylabel, log_y) in enumerate(active):
        ax_prof  = axes[row, 0]
        ax_ratio = axes[row, 1]

        r_ref, p_ref = sims[ref_label]['profiles'][field][snap_num]

        for label in labels_list:
            r, p = sims[label]['profiles'][field][snap_num]
            if r is None:
                continue
            
            ls = RUNS[label]['ls']
            c  = RUNS[label]['c']
            
            ax_prof.plot(r, p, color=c, lw=2.0, alpha=0.9, ls=ls, label=label)

            if r_ref is not None and label != ref_label:
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Interpolate reference profile exactly to this r or assume same r
                    ratio = p / p_ref
                ax_ratio.plot(r, ratio, color=c, lw=1.8, ls=ls, label=f'{label} / {ref_label}')

        ax_prof.set_ylabel(ylabel, fontsize=10)
        ax_prof.set_xlabel('Radius [kpc]', fontsize=9)
        if log_y:
            ax_prof.set_yscale('log')
        ax_prof.legend(fontsize=8, loc='best', framealpha=0.7)

        ax_ratio.axhline(1.0, color='k', lw=0.9, ls='-', alpha=0.7)
        ax_ratio.set_ylabel('Ratio', fontsize=9)
        ax_ratio.set_xlabel('Radius [kpc]', fontsize=9)
        ax_ratio.set_title(field, fontsize=9, pad=3)
        if ax_ratio.has_data():
            ax_ratio.legend(fontsize=8)

    plt.tight_layout()
    fname = os.path.join(OUTDIR, f'radial_compare_snap{snap_num:03d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Snap {snap_num:03d} -> {fname}')

print('\nDone.')
