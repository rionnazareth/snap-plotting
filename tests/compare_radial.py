"""
compare_radial.py
=================
1D radial profile comparison of:
    output_cr     (no CR diffusion)  — blue tones
    output_crexps (with CR diffusion) — orange tones

Mirrors the style of radial_plots.py (linear radial bins, publication-quality
formatting) but overlays both simulations on the same axis and adds a ratio
panel so the diffusion effect is quantified at every radius.

For every shared snapshot this script produces one PNG with:
  - Left column  : actual profiles   (no diffusion  vs  with diffusion)
  - Right column : ratio             (with diff / no diff)

A final summary PNG overlays all snapshots per quantity on a 3×2 grid.

Saved to: test_ai/plots/
"""

import sys, os
sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/arepo-snap-util')
sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/gasCloudNfw/snap-plotting')

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

PATH_CR    = BASE_PATH + '/output_cr/'
PATH_CREXP = BASE_PATH + '/output_crexps/'
PATH_NOCR  = BASE_PATH + '/output2/'
N_CR       = 7
N_CREXP    = 7
N_NOCR     = 7
COMMON_N   = min(N_CR, N_CREXP, N_NOCR)

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(OUTDIR, exist_ok=True)

R_RANGE = (3.0, 30.0)
NBINS   = 500

# ── quantity catalogue ────────────────────────────────────────────────────────
# (field, y-label, log_y, colour_cr, colour_crex, colour_nocr)
QUANTITIES = [
    ('rho',       r'Density $\rho$ [code]',      True,  '#2166ac', '#d6604d', '#4d4d4d'),
    ('pres',      r'Thermal pressure [code]',     True,  '#4393c3', '#f4a582', '#878787'),
    ('crpres',    r'CR pressure [code]',          True,  '#006837', '#d9ef8b', '#bababa'),
    ('cren',      r'CR energy density [code]',    True,  '#1a9850', '#fee08b', '#e0e0e0'),
    ('u',         r'Internal energy [code]',      True,  '#762a83', '#c2a5cf', '#4d4d4d'),
    ('speed',     r'Speed |v| [code km/s]',       True,  '#b2182b', '#ef8a62', '#878787'),
    ('mach',      r'Mach number',                 True,  '#543005', '#bf812d', '#bababa'),
    ('vrad',      r'Radial velocity [code km/s]', False, '#313695', '#fdae61', '#e0e0e0'),
    ('bflden',    r'B-field energy density [code]', True, '#000004', '#9970ab', '#4d4d4d')
]

# ── helpers ───────────────────────────────────────────────────────────────────
def snap_time_myr(s):
    Lu = s.header['UnitLength_in_cm'] * u.cm
    Vu = s.header['UnitVelocity_in_cm_per_s'] * u.cm / u.s
    return (s.header['Time'] * Lu / Vu).to(u.Myr).value


def enrich_snap(s):
    unit_l = s.header['UnitLength_in_cm']
    unit_v = s.header['UnitVelocity_in_cm_per_s']
    unit_rho = s.header['UnitMass_in_g'] / unit_l**3

    mu = 0.6 * m_p
    s.data['temp']  = (GAMMA - 1) * mu / k_B * s.data['u'] * unit_v**2
    s.data['speed'] = np.linalg.norm(s.data['vel'], axis=1)
    if 'cren' in s.data:
        s.data['crpres'] = (GAMMA_CR - 1) * s.data['rho'] * s.data['cren']

    unit_b = np.sqrt(4 * np.pi * unit_rho * unit_v**2)

    if 'bfld' in s.data:
        # B-field specific energy (energy per unit mass in code units -> cgs)
        b_cgs = s.data['bfld'] * unit_b
        rho_cgs = s.data['rho'] * unit_rho
        s.data['bflden'] = np.sum(b_cgs**2, axis=1) / (8 * np.pi * rho_cgs)
        s.data['bflds'] = np.linalg.norm(s.data['bfld'], axis=1)
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

snaps_cr, snaps_crex, snaps_nocr  = [], [], []
times_cr, times_crex, times_nocr  = [], [], []

for n in range(COMMON_N):
    oc = arun.Run(snappath=PATH_CR,    snapbase=SNAPBASE)
    oe = arun.Run(snappath=PATH_CREXP, snapbase=SNAPBASE)
    on = arun.Run(snappath=PATH_NOCR,  snapbase=SNAPBASE)
    sc = enrich_snap(oc.loadSnap(snapnum=n))
    se = enrich_snap(oe.loadSnap(snapnum=n))
    sn = enrich_snap(on.loadSnap(snapnum=n))
    snaps_cr.append(sc);   times_cr.append(snap_time_myr(sc))
    snaps_crex.append(se); times_crex.append(snap_time_myr(se))
    snaps_nocr.append(sn); times_nocr.append(snap_time_myr(sn))
    print(f'  snap {n:03d} — cr: {times_cr[-1]:.1f} Myr | crexps: {times_crex[-1]:.1f} Myr | nocr: {times_nocr[-1]:.1f} Myr')

# ── pre-compute all profiles ──────────────────────────────────────────────────
# profiles[run_key][field][snap_num] = (r_ctrs, profile_array)
profiles = {'cr': {}, 'crex': {}, 'nocr': {}}
for field, *_ in QUANTITIES:
    profiles['cr'][field]   = [radial_profile_lin(s, field) for s in snaps_cr]
    profiles['crex'][field] = [radial_profile_lin(s, field) for s in snaps_crex]
    profiles['nocr'][field] = [radial_profile_lin(s, field) for s in snaps_nocr]

# ── per-snapshot figures ──────────────────────────────────────────────────────
for snap_num in range(COMMON_N):
    t_cr   = times_cr[snap_num]
    t_crex = times_crex[snap_num]
    t_nocr = times_nocr[snap_num]

    # Keep only fields available in at least one run
    active = [(f, lbl, lg, c1, c2, c3)
              for f, lbl, lg, c1, c2, c3 in QUANTITIES
              if profiles['cr'][f][snap_num][0] is not None
              or profiles['crex'][f][snap_num][0] is not None
              or profiles['nocr'][f][snap_num][0] is not None]

    if not active:
        print(f'Snap {snap_num:03d}: no plottable fields, skipping.')
        continue

    n_qty = len(active)
    fig, axes = plt.subplots(n_qty, 2, figsize=(13, 3.8 * n_qty), squeeze=False)

    fig.suptitle(
        f'Radial profiles — snap {snap_num:03d}\n'
        f'No diff: {t_cr:.1f} Myr   |   With diff: {t_crex:.1f} Myr   |   No CRs: {t_nocr:.1f} Myr',
        fontsize=12, y=1.002,
    )
    axes[0, 0].set_title('Profiles (solid=no diff, dashed=with diff, dotted=no CRs)', fontsize=10)
    axes[0, 1].set_title('Ratio to no diff (output_cr)', fontsize=10)

    for row, (field, ylabel, log_y, col_cr, col_crex, col_nocr) in enumerate(active):
        r_cr,   p_cr   = profiles['cr'][field][snap_num]
        r_crex, p_crex = profiles['crex'][field][snap_num]
        r_nocr, p_nocr = profiles['nocr'][field][snap_num]

        ax_prof  = axes[row, 0]
        ax_ratio = axes[row, 1]

        # ---- left: overlay profiles ----
        if r_cr is not None:
            ax_prof.plot(r_cr,  p_cr,   color=col_cr,   lw=2.0, alpha=0.9,
                         ls='-',  label='no diff')
        if r_crex is not None:
            ax_prof.plot(r_crex, p_crex, color=col_crex, lw=2.0, alpha=0.9,
                         ls='--', label='with diff')
        if r_nocr is not None:
            ax_prof.plot(r_nocr, p_nocr, color=col_nocr, lw=2.0, alpha=0.9,
                         ls=':', label='no CRs')
        ax_prof.set_ylabel(ylabel, fontsize=10)
        ax_prof.set_xlabel('Radius [kpc]', fontsize=9)
        if log_y:
            ax_prof.set_yscale('log')
        ax_prof.legend(fontsize=8, loc='best', framealpha=0.7)

        # ---- right: ratio ----
        if r_cr is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                if r_crex is not None:
                    ratio_crex = p_crex / p_cr
                    ax_ratio.plot(r_cr, ratio_crex, color='darkorchid', lw=1.8, ls='--', label='with diff / no diff')
                if r_nocr is not None:
                    ratio_nocr = p_nocr / p_cr
                    ax_ratio.plot(r_cr, ratio_nocr, color='dimgrey', lw=1.8, ls=':', label='no CRs / no diff')
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
        r_cr,   p_cr   = profiles['cr'][field][snap_num]
        r_crex, p_crex = profiles['crex'][field][snap_num]
        r_nocr, p_nocr = profiles['nocr'][field][snap_num]

        show_label = ax_idx == 0   # only label the first panel
        if r_cr is not None:
            ax.plot(r_cr,  p_cr,   color=color, lw=1.8, alpha=0.85, ls='-',
                    label=f'no-diff s{snap_num}' if show_label else None)
        if r_crex is not None:
            ax.plot(r_crex, p_crex, color=color, lw=1.8, alpha=0.85, ls='--',
                    label=f'w-diff  s{snap_num}' if show_label else None)
        if r_nocr is not None:
            ax.plot(r_nocr, p_nocr, color=color, lw=1.8, alpha=0.85, ls=':',
                    label=f'no-CRs  s{snap_num}' if show_label else None)

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
    Line2D([0], [0], color='grey', lw=2, ls='-',  label='no diffusion (output_cr)'),
    Line2D([0], [0], color='grey', lw=2, ls='--', label='with diffusion (output_crexps)'),
    Line2D([0], [0], color='grey', lw=2, ls=':',  label='no CRs (output2)'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle('All snapshots: solid = no diffusion, dashed = with diffusion, dotted = no CRs', fontsize=12)
plt.tight_layout()

fname = os.path.join(OUTDIR, 'radial_summary_all_snaps.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Summary -> {fname}')
print('\nDone.')
