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
N_CR       = 7    # snaps 000–006
N_CREXP    = 7    # snaps 000–006
N_NOCR     = 7    # snaps 000–006
COMMON_N   = min(N_CR, N_CREXP, N_NOCR)   # 6

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(OUTDIR, exist_ok=True)

# ── helper functions ──────────────────────────────────────────────────────────
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
    return s


def radial_profile_log(s, field, r_range=(2.5, 200), nbins=200):
    """
    Logarithmically binned radial profile.  Returns (r_centres, values).
    Mirrors rvsval() from shocks1d.ipynb.
    """
    pos = s.data['pos']
    ctr = np.array([s.boxsize / 2] * 3)
    r   = np.linalg.norm(pos - ctr, axis=1)

    # radial velocity (attach once)
    if 'vrad' not in s.data:
        diff = pos - ctr
        vdot = np.sum(diff * s.data['vel'], axis=1)
        s.data['vrad'] = vdot / (r + 1e-30)

    if field not in s.data:
        return None, None

    vals     = s.data[field]
    rlo, rhi = r_range
    mask     = (r >= rlo) & (r <= rhi) & np.isfinite(vals)

    r_bins   = np.logspace(np.log10(rlo), np.log10(rhi), nbins + 1)
    r_ctrs   = 0.5 * (r_bins[:-1] + r_bins[1:])
    idx      = np.digitize(r[mask], r_bins)
    profile  = np.array([
        vals[mask][idx == i].mean() if np.any(idx == i) else np.nan
        for i in range(1, nbins + 1)
    ])
    return r_ctrs, profile


# ── quantity catalogue ────────────────────────────────────────────────────────
# (field_key,   label,                   log_y?)
QUANTITIES = [
    ('rho',    r'Density $\rho$ [code]',          True),
    ('pres',   r'Thermal pressure [code]',         True),
    ('crpres', r'CR pressure [code]',              True),
    ('u',      r'Internal energy [code]',          True),
    ('speed',  r'Speed [code km/s]',               True),
    ('mach',   r'Mach number',                     True),
]

R_RANGE = (11.0, 50.0)   # kpc
NBINS   = 300

# ── load ALL snapshots for both runs ─────────────────────────────────────────
print('Loading all snapshots...')

snaps_cr, snaps_crex, snaps_nocr = [], [], []
times_cr, times_crex, times_nocr = [], [], []

for n in range(N_CR):
    o = arun.Run(snappath=PATH_CR, snapbase=SNAPBASE)
    s = enrich_snap(o.loadSnap(snapnum=n))
    snaps_cr.append(s)
    times_cr.append(snap_time_myr(s))
    print(f'  output_cr   snap {n:03d}  {times_cr[-1]:.1f} Myr')

for n in range(N_CREXP):
    o = arun.Run(snappath=PATH_CREXP, snapbase=SNAPBASE)
    s = enrich_snap(o.loadSnap(snapnum=n))
    snaps_crex.append(s)
    times_crex.append(snap_time_myr(s))
    print(f'  output_crexps snap {n:03d}  {times_crex[-1]:.1f} Myr')

for n in range(N_NOCR):
    o = arun.Run(snappath=PATH_NOCR, snapbase=SNAPBASE)
    s = enrich_snap(o.loadSnap(snapnum=n))
    snaps_nocr.append(s)
    times_nocr.append(snap_time_myr(s))
    print(f'  output2     snap {n:03d}  {times_nocr[-1]:.1f} Myr')

# ── one combined figure per quantity ─────────────────────────────────────────
for field, label, log_y in QUANTITIES:

    # Check at least one snapshot has this field
    has_field = any(field in s.data for s in snaps_cr + snaps_crex + snaps_nocr)
    if not has_field:
        print(f'  Skipping {field}: not present in any snapshot')
        continue

    print(f'\nPlotting evolution: {field}')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    ax_cr, ax_crex, ax_diff = axes

    # colour maps: early → dark, late → bright
    cmap_cr   = cm.get_cmap('Blues_r',  N_CR)
    cmap_crex = cm.get_cmap('Oranges_r', N_CREXP)
    cmap_nocr = cm.get_cmap('Greys_r', N_NOCR)

    # ---- left panel: output_cr evolution ----
    r_last_cr, prof_last_cr = None, None
    for n, s in enumerate(snaps_cr):
        r, prof = radial_profile_log(s, field, r_range=R_RANGE, nbins=NBINS)
        if r is None:
            continue
        color = cmap_cr(n / max(N_CR - 1, 1))
        lw    = 1.5 + 0.5 * (n / (N_CR - 1))
        ax_cr.plot(r, prof, color=color, lw=lw, alpha=0.9,
                   label=f'snap {n:02d}  ({times_cr[n]:.0f} Myr)')
        if n == COMMON_N - 1:
            r_last_cr, prof_last_cr = r, prof

    for n, s in enumerate(snaps_nocr):
        r, prof = radial_profile_log(s, field, r_range=R_RANGE, nbins=NBINS)
        if r is None:
            continue
        color = cmap_nocr(n / max(N_NOCR - 1, 1))
        lw    = 1.5 + 0.5 * (n / (N_NOCR - 1))
        ax_cr.plot(r, prof, color=color, lw=lw, alpha=0.7, ls='--',
                   label=f'no-CR (output2)' if n == COMMON_N - 1 else "")

    ax_cr.set_title('output_cr  (no diffusion)', fontsize=11)
    ax_cr.set_xlabel('Radius [kpc]', fontsize=10)
    ax_cr.set_ylabel(label, fontsize=10)
    ax_cr.set_xscale('log')
    if log_y:
        ax_cr.set_yscale('log')
    ax_cr.legend(fontsize=7, loc='best', framealpha=0.6)
    ax_cr.grid(True, which='both', alpha=0.25, ls='--')

    # ---- middle panel: output_crexps evolution ----
    r_last_crex, prof_last_crex = None, None
    for n, s in enumerate(snaps_crex):
        r, prof = radial_profile_log(s, field, r_range=R_RANGE, nbins=NBINS)
        if r is None:
            continue
        color = cmap_crex(n / max(N_CREXP - 1, 1))
        lw    = 1.5 + 0.5 * (n / (N_CREXP - 1))
        ax_crex.plot(r, prof, color=color, lw=lw, alpha=0.9,
                     label=f'snap {n:02d}  ({times_crex[n]:.0f} Myr)')
        if n == COMMON_N - 1:
            r_last_crex, prof_last_crex = r, prof

    for n, s in enumerate(snaps_nocr):
        r, prof = radial_profile_log(s, field, r_range=R_RANGE, nbins=NBINS)
        if r is None:
            continue
        color = cmap_nocr(n / max(N_NOCR - 1, 1))
        lw    = 1.5 + 0.5 * (n / (N_NOCR - 1))
        ax_crex.plot(r, prof, color=color, lw=lw, alpha=0.7, ls='--',
                     label=f'no-CR (output2)' if n == COMMON_N - 1 else "")

    ax_crex.set_title('output_crexps  (with diffusion)', fontsize=11)
    ax_crex.set_xlabel('Radius [kpc]', fontsize=10)
    ax_crex.set_xscale('log')
    if log_y:
        ax_crex.set_yscale('log')
    ax_crex.legend(fontsize=7, loc='best', framealpha=0.6)
    ax_crex.grid(True, which='both', alpha=0.25, ls='--')

    # ---- right panel: relative difference at last common snapshot ----
    if r_last_cr is not None and r_last_crex is not None:
        # Both profiles use the same r grid (same NBINS, same R_RANGE)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = (prof_last_crex - prof_last_cr) / np.abs(prof_last_cr)

        ax_diff.plot(r_last_cr, rel, color='purple', lw=2)
        ax_diff.axhline(0, color='k', lw=0.8, ls='--')
        ax_diff.set_title(
            f'Relative diff at snap {COMMON_N-1:02d}\n'
            r'$(crexps - cr)\,/\,|cr|$', fontsize=10)
        ax_diff.set_xlabel('Radius [kpc]', fontsize=10)
        ax_diff.set_ylabel('Relative difference', fontsize=10)
        ax_diff.set_xscale('log')
        ax_diff.grid(True, which='both', alpha=0.25, ls='--')

    else:
        ax_diff.text(0.5, 0.5, 'Data unavailable',
                     ha='center', va='center', transform=ax_diff.transAxes)
        ax_diff.set_axis_off()

    fig.suptitle(f'Evolution comparison — {label}', fontsize=13)
    plt.tight_layout()

    fname = os.path.join(OUTDIR, f'evolution_{field}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {fname}')

# ── bonus: shock-front radius vs time ─────────────────────────────────────────
# Track position of Mach-number peak across snapshots for both runs
print('\nPlotting shock radius evolution...')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_r, ax_log = axes

def find_shock_radius(snaps, times, r_range, nbins):
    r_shock, t_arr = [], []
    for n, (s, t) in enumerate(zip(snaps, times)):
        r, mach = radial_profile_log(s, 'mach', r_range=r_range, nbins=nbins)
        if r is None or not np.any(np.isfinite(mach)):
            continue
        idx = np.nanargmax(mach)
        r_shock.append(r[idx])
        t_arr.append(t)
    return np.array(t_arr), np.array(r_shock)

t_cr_arr,  r_cr_arr  = find_shock_radius(snaps_cr,   times_cr,   R_RANGE, NBINS)
t_crex_arr, r_crex_arr = find_shock_radius(snaps_crex, times_crex, R_RANGE, NBINS)
t_nocr_arr, r_nocr_arr = find_shock_radius(snaps_nocr, times_nocr, R_RANGE, NBINS)

for ax in axes:
    ax.plot(t_cr_arr,   r_cr_arr,   'o-', color='steelblue',
            lw=2, label='no diffusion (output_cr)')
    ax.plot(t_crex_arr, r_crex_arr, 's--', color='darkorange',
            lw=2, label='with diffusion (output_crexps)')
    ax.plot(t_nocr_arr, r_nocr_arr, '^:', color='dimgrey',
            lw=2, label='no CRs (output2)')
    ax.set_xlabel('Time [Myr]', fontsize=11)
    ax.set_ylabel('Shock radius [kpc]', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, ls='--')

# Add power-law reference lines to the log-log panel
ax_log.set_xscale('log')
ax_log.set_yscale('log')
if len(t_cr_arr) > 1 and np.any(t_cr_arr > 0):
    idx0   = np.argmax(t_cr_arr > 0)   # first snapshot with t > 0
    t0, r0 = t_cr_arr[idx0], r_cr_arr[idx0]
    t_max  = max([t[-1] for t in (t_cr_arr, t_crex_arr, t_nocr_arr) if len(t) > 0])
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
