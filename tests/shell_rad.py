import sys, os
sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/arepo-snap-util')
sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/gasCloudNfw/snap-plotting')

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from lib import *

plt.style.use(['science'])

BASE = '/cosma8/data/dp317/dc-naza3/homogeneous'
RUNS = {
    r'$n_\mathrm{H}=50$ cm$^{-3}$': {'path': BASE + '/et_backup/50/',  'ls': '-',  'c': 'C0'},
    r'$n_\mathrm{H}=5$ cm$^{-3}$':           {'path': BASE + '/et_backup/5/',   'ls': '--', 'c': 'C1'},
    r'$n_\mathrm{H}=0.5$ cm$^{-3}$':        {'path': BASE + '/et_backup/0.5/', 'ls': ':',  'c': 'C2'},
    # r'hydro run': {'path': BASE + '/mtests/output_bf/', 'ls': '-', 'c': 'C3'},
    # r'with CRs':    {'path': BASE + '/mtests/output_bfcr/', 'ls': '--', 'c': 'C4'},
}
SNAPBASE = 'snap_'
N_SNAPS  = 9
OUTDIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rad')
os.makedirs(OUTDIR, exist_ok=True)

# R_free parameters
L_AGN = 1e45        # erg/s
BETA  = 1e4 / 3e5   # v_w / c
TAU   = 1.0
B     = 1.0

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5))

ref_anchored = False
for label, info in RUNS.items():
    t_arr, r_lower, r_upper = [], [], []

    for n in range(N_SNAPS):
        try:
            s = load_snap_data(n, snappath=info['path'] + '/', snapbase=SNAPBASE)
            r_l, r_u = find_shell_radius(s)
            if not (np.isnan(r_l) or np.isnan(r_u)):
                t_arr.append(calc_snap_time(s))
                r_lower.append(r_l * 1e3)   # kpc → pc
                r_upper.append(r_u * 1e3)
        except Exception as e:
            print(f'  {label} snap {n:03d}: {e}')

    if not t_arr:
        continue

    t_arr   = np.array(t_arr)
    r_lower = np.array(r_lower)
    r_upper = np.array(r_upper)
    r_mid   = 0.5 * (r_lower + r_upper)

    ax.fill_between(t_arr, r_lower, r_upper, color=info['c'], alpha=0.3, lw=0)
    ax.plot(t_arr, r_mid, color=info['c'], ls=info['ls'], lw=2, label=label)

    # Local log-log slope: d(log R)/d(log t)
    slope = np.gradient(np.log(r_mid), np.log(t_arr))
    ax2.plot(t_arr, slope, color=info['c'], ls=info['ls'], lw=2, label=label)

    # Anchor reference lines to first run
    if not ref_anchored:
        t0, r0 = t_arr[0], r_mid[0]
        t_ref  = np.logspace(np.log10(t_arr.min()), np.log10(t_arr.max()), 300)
        ax.plot(t_ref, r0 * (t_ref / t0)**1,
                'k--', lw=1, label=r'$\propto t$')
        ax.plot(t_ref, r0 * (t_ref / t0)**(3/5),
                color='steelblue', ls=':', lw=1, label=r'$\propto t^{3/5}$')
        ax2.axhline(1,   color='k',          ls='--', lw=1, label=r'$\alpha=1$')
        ax2.axhline(3/5, color='steelblue',  ls=':',  lw=1, label=r'$\alpha=3/5$')
        ref_anchored = True

    # R_free horizontal line
    try:
        s0    = load_snap_data(0, snappath=info['path'] + '/', snapbase=SNAPBASE)
        n0    = np.nanmedian(s0.data['n_dens_cm'])
        rho_0 = n0 * m_p * 0.6
        r_free_pc, _ = calculate_R_free(BETA, TAU, B, L_AGN, rho_0=rho_0)
        ax.axhline(r_free_pc, color=info['c'], ls='--', lw=1, alpha=0.8,
                   label=rf'$R_{{\rm free}}$ ({label})')
        print(f'  {label}: R_free = {r_free_pc:.1f} pc  (n0 = {n0:.2e} cm^-3)')
    except Exception as e:
        print(f'  R_free for {label} failed: {e}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$t\ [\mathrm{Myr}]$')
ax.set_ylabel(r'$R\ [\mathrm{pc}]$')
ax.legend(fontsize=7, loc='upper left', framealpha=0.5)

ax2.set_xscale('log')
ax2.set_xlabel(r'$t\ [\mathrm{Myr}]$')
ax2.set_ylabel(r'$\alpha = \mathrm{d}\log R\,/\,\mathrm{d}\log t$')
ax2.legend(fontsize=7, loc='upper right', framealpha=0.5)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'shell_radius_rfree.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n-> Saved {fname}')