"""
compare_radial.py
=================
1D radial profile comparison of:
    output_tree  (tree based)  — blue tones
    output_et    (equal timesteps)    — grey tones

Mirrors the style of radial_plots.py (linear radial bins, publication-quality
formatting) but overlays both simulations on the same axis and adds a ratio
panel so the effect is quantified at every radius.

For every shared snapshot this script produces one PNG with:
  - Left column  : actual profiles   (tree based  vs  equal timesteps)
  - Right column : ratio             (equal timesteps / tree based)

A final summary PNG overlays all snapshots per quantity on a 3×2 grid.

Saved to: test_ai/plots/
"""

import sys, os
sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/arepo-snap-util')
# sys.path.insert(0, '/home/c5046973/agn/gasCloudTest/arepo_t/snap-plotting')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import astropy.units as u
import arepo_run as arun

# ── constants & paths ─────────────────────────────────────────────────────────
BASE_PATH  = '/cosma8/data/dp317/dc-naza3/gasCloudNfw'
SNAPBASE   = 'snap_'
GAMMA      = 5./3
GAMMA_CR   = 4./3
k_B        = 1.381e-16
m_p        = 1.66e-24
unit_v     = 1.e5

PATH_TREE  = BASE_PATH + '/output2/'
PATH_ET    = BASE_PATH + '/output_maxt/'
N_TREE     = 11
N_ET       = 4
COMMON_N   = min(N_TREE, N_ET)

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compare_maxt')
os.makedirs(OUTDIR, exist_ok=True)

R_RANGE = (0, 30)
NBINS   = 500

# ── quantity catalogue ────────────────────────────────────────────────────────
# (field, y-label, log_y, colour_tree, colour_et)
QUANTITIES = [
    ('rho',       r'Density $\rho$ [code]',      True,  '#2166ac', '#4d4d4d'),
    ('pres',      r'Thermal pressure [code]',     True,  '#4393c3', '#878787'),
    ('crpres',    r'CR pressure [code]',          True,  '#006837', '#bababa'),
    ('cren',      r'CR energy density [code]',    True,  '#1a9850', '#e0e0e0'),
    ('u',         r'Internal energy [code]',      True,  '#762a83', '#4d4d4d'),
    ('speed',     r'Speed |v| [code km/s]',       True,  '#b2182b', '#878787'),
    ('mach',      r'Mach number',                 True,  '#543005', '#bababa'),
    ('vrad',      r'Radial velocity [code km/s]', False, '#313695', '#e0e0e0'),
]

# ── helpers ───────────────────────────────────────────────────────────────────
def snap_time_myr(s):
    Lu = s.header['UnitLength_in_cm'] * u.cm
    Vu = s.header['UnitVelocity_in_cm_per_s'] * u.cm / u.s
    return (s.header['Time'] * Lu / Vu).to(u.Myr).value


def enrich_snap(s):
    mu = 0.6 * m_p
    s.data['temp']  = (GAMMA - 1) * mu / k_B * s.data['u'] * unit_v**2
    s.data['speed'] = np.linalg.norm(s.data['vel'], axis=1)
    if 'cren' in s.data:
        s.data['crpres'] = (GAMMA_CR - 1) * s.data['rho'] * s.data['cren']
    if 'bfld' in s.data:
        s.data['bfldenerg'] = np.sum(s.data['bfld']**2, axis=1) / (2 * s.data['rho'])
    pos  = s.data['pos']
    ctr  = np.array([s.boxsize / 2] * 3)
    diff = pos - ctr
    r    = np.linalg.norm(diff, axis=1)
    s.data['vrad'] = np.sum(diff * s.data['vel'], axis=1) / (r + 1e-30)
    return s


def radial_profile_lin(s, field, r_range=R_RANGE, nbins=NBINS):
    """Linearly binned radial profile.  Returns (r_centres, profile) or (None, None)."""
    if field not in s.data:
        return None, None
    pos  = s.data['pos']
    ctr  = np.array([s.boxsize / 2] * 3)
    r    = np.linalg.norm(pos - ctr, axis=1)
    vals = s.data[field]
    rlo, rhi = r_range
    mask = (r >= rlo) & (r <= rhi) & np.isfinite(vals)
    r_bins  = np.linspace(rlo, rhi, nbins + 1)
    r_ctrs  = 0.5 * (r_bins[:-1] + r_bins[1:])
    sums, _ = np.histogram(r[mask], bins=r_bins, weights=vals[mask])
    cnts, _ = np.histogram(r[mask], bins=r_bins)
    with np.errstate(invalid='ignore'):
        profile = np.where(cnts > 0, sums / cnts, np.nan)
    return r_ctrs, profile


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


# ── load all snapshots ONCE ───────────────────────────────────────────────────
apply_style()
print('Loading all snapshots (this may take a minute) ...')

snaps_tree, snaps_et  = [], []
times_tree, times_et  = [], []

for n in range(COMMON_N):
    oc = arun.Run(snappath=PATH_TREE,   snapbase=SNAPBASE)
    on = arun.Run(snappath=PATH_ET, snapbase=SNAPBASE)
    sc = enrich_snap(oc.loadSnap(snapnum=n))
    sn = enrich_snap(on.loadSnap(snapnum=n))
    snaps_tree.append(sc);   times_tree.append(snap_time_myr(sc))
    snaps_et.append(sn); times_et.append(snap_time_myr(sn))
    print(f'  snap {n:03d} — tree: {snap_time_myr(sc):.1f} Myr | et: {snap_time_myr(sn):.1f} Myr')

# ── pre-compute all profiles ──────────────────────────────────────────────────
# profiles[run_key][field][snap_num] = (r_ctrs, profile_array)
profiles = {'tree': {}, 'et': {}}
for field, *_ in QUANTITIES:
    profiles['tree'][field]   = [radial_profile_lin(s, field) for s in snaps_tree]
    profiles['et'][field] = [radial_profile_lin(s, field) for s in snaps_et]

# ── per-snapshot figures ──────────────────────────────────────────────────────
for snap_num in range(COMMON_N):
    t_tree   = times_tree[snap_num]
    t_et = times_et[snap_num]

    # Keep only fields available in at least one run
    active = [(f, lbl, lg, c1, c2)
              for f, lbl, lg, c1, c2 in QUANTITIES
              if profiles['tree'][f][snap_num][0] is not None
              or profiles['et'][f][snap_num][0] is not None]

    if not active:
        print(f'Snap {snap_num:03d}: no plottable fields, skipping.')
        continue

    n_qty = len(active)
    fig, axes = plt.subplots(n_qty, 2, figsize=(13, 3.8 * n_qty), squeeze=False)

    fig.suptitle(
        f'Radial profiles — snap {snap_num:03d}\n'
        f'MaxSizeTimestep0.01: {t_tree:.1f} Myr   |   MaxSizeTimestep0.0001: {t_et:.1f} Myr',
        fontsize=12, y=1.002,
    )
    axes[0, 0].set_title('Profiles (solid=tree based, dashed=equal timesteps)', fontsize=10)
    axes[0, 1].set_title('Ratio: equal timesteps / tree based', fontsize=10)

    for row, (field, ylabel, log_y, col_tree, col_et) in enumerate(active):
        r_tree,   p_tree   = profiles['tree'][field][snap_num]
        r_et, p_et = profiles['et'][field][snap_num]

        ax_prof  = axes[row, 0]
        ax_ratio = axes[row, 1]

        # ---- left: overlay profiles ----
        if r_tree is not None:
            ax_prof.plot(r_tree,   p_tree,   color=col_tree,   lw=2.0, alpha=0.9,
                         ls='-',  label='MaxSizeTimestep0.01')
        if r_et is not None:
            ax_prof.plot(r_et, p_et, color=col_et, lw=2.0, alpha=0.9,
                         ls='--', label='MaxSizeTimestep0.0001')
        ax_prof.set_ylabel(ylabel, fontsize=10)
        ax_prof.set_xlabel('Radius [kpc]', fontsize=9)
        if log_y:
            ax_prof.set_yscale('log')
        ax_prof.legend(fontsize=8, loc='best', framealpha=0.7)

        # ---- right: ratio ----
        if r_tree is not None and r_et is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_et = p_et / p_tree
            ax_ratio.plot(r_tree, ratio_et, color='dimgrey', lw=1.8, ls='--', label='MaxSizeTimestep0.0001 / MaxSizeTimestep0.01')
            ax_ratio.axhline(1.0, color='k', lw=0.9, ls='-', alpha=0.7, label='ratio = 1')
            ax_ratio.set_ylabel('Ratio', fontsize=9)
            ax_ratio.legend(fontsize=8)
        else:
            ax_ratio.text(0.5, 0.5, 'N/A', ha='center', va='center',
                          transform=ax_ratio.transAxes)
        ax_ratio.set_xlabel('Radius [kpc]', fontsize=9)
        ax_ratio.set_title(field, fontsize=9, pad=3)

    plt.tight_layout()
    fname = os.path.join(OUTDIR, f'radial_compare_snap{snap_num:03d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Snap {snap_num:03d} -> {fname}')

# ── summary figure: all snapshots on one 3×2 page ────────────────────────────
print('\nBuilding summary figure ...')

SUMMARY_QTY = [
    ('rho',    r'$\rho$',              True),
    ('pres',   r'$P_\mathrm{th}$',     True),
    ('crpres', r'$P_\mathrm{CR}$',     True),
    ('cren',   r'$e_\mathrm{CR}$',     True),
    ('speed',  r'$|v|$',               True),
    ('mach',   r'$\mathcal{M}$',       True),
]

cmap_snaps = plt.cm.get_cmap('plasma', COMMON_N)
ncols, nrows = 3, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.8), squeeze=False)
axes_flat = axes.flatten()

for ax_idx, (field, short_label, log_y) in enumerate(SUMMARY_QTY):
    ax = axes_flat[ax_idx]
    for snap_num in range(COMMON_N):
        color = cmap_snaps(snap_num / max(COMMON_N - 1, 1))
        r_tree,   p_tree   = profiles['tree'][field][snap_num]
        r_et, p_et = profiles['et'][field][snap_num]

        show_label = ax_idx == 0   # only label the first panel
        if r_tree is not None:
            ax.plot(r_tree,   p_tree,   color=color, lw=1.8, alpha=0.85, ls='-',
                    label=f'MaxSizeTimestep0.01 s{snap_num}' if show_label else None)
        if r_et is not None:
            ax.plot(r_et, p_et, color=color, lw=1.8, alpha=0.85, ls='--',
                    label=f'MaxSizeTimestep0.0001 s{snap_num}' if show_label else None)

    ax.set_title(short_label, fontsize=12)
    ax.set_xlabel('r [kpc]', fontsize=9)
    ax.set_ylabel(short_label, fontsize=9)
    if log_y:
        ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.2, ls='--')

# Legend on first panel
axes_flat[0].legend(fontsize=6, loc='best', framealpha=0.5, ncol=2)

# Snapshot colour bar
sm = plt.cm.ScalarMappable(cmap=cmap_snaps,
                            norm=plt.Normalize(vmin=0, vmax=COMMON_N - 1))
sm.set_array([])
# cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.45, pad=0.02)
# cbar.set_label('Snapshot index', fontsize=10)
# cbar.set_ticks(range(COMMON_N))

# Line-style legend
handles = [
    Line2D([0], [0], color='grey', lw=2, ls='-',  label='MaxSizeTimestep0.01'),
    Line2D([0], [0], color='grey', lw=2, ls='--', label='MaxSizeTimestep0.0001'),
]
fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle('All snapshots: solid = MaxSizeTimestep0.01, dashed = MaxSizeTimestep0.0001', fontsize=12)
plt.tight_layout()

fname = os.path.join(OUTDIR, 'radial_summary_all_snaps.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Summary -> {fname}')
print('\nDone.')
