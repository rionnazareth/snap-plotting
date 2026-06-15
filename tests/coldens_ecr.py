#!/cosma/apps/dp317/dc-naza3/renv/bin/python
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/snap-plotting/tests')
from lib import *

# ── simulation paths ──────────────────────────────────────────────────────────
BASE     = '/cosma8/data/dp317/dc-naza3/homogeneous/et_backup'
SNAPBASE = 'snap_'

RUNS = {
    r'$\rho = \rho_0 / 10$':      {'path': BASE + '/0.5/', 'c': 'C1', 'marker': 's'},
    r'$\rho = \rho_0$':           {'path': BASE + '/5/',   'c': 'C2', 'marker': 'D'},
    r'$\rho = \rho_0 \times 10$': {'path': BASE + '/50/',  'c': 'C0', 'marker': 'o'},
}

SNAP_NUMS  = [3, 7, 13]
SNAP_ALPHA = [0.5, 0.75, 1.0]   # older → more transparent
SNAP_SIZE  = [60,  100,  160]    # older → smaller

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(OUTDIR, exist_ok=True)

# ── physical constants ────────────────────────────────────────────────────────
L_AGN       = 1e45          # erg s^-1
v_w         = 1e4 * 1e5     # cm s^-1  (10^4 km/s)
c_light     = 3e10          # cm s^-1
beta        = v_w / c_light
E_w_dot     = 0.5 * beta * L_AGN   # wind power (erg s^-1)
myr_to_s    = 1e6 * 365.25 * 24 * 3600
X_H         = 0.76          # hydrogen mass fraction
m_p         = 1.67e-24      # g


def column_density(s, unit_m, unit_l):
    """
    Mean column density through the wind shell [cm^-2].
    N_H = M_shell_H / (m_p * 4pi * R_sh^2)
    where R_sh is the 99.7th-percentile radius of wind tracer cells.
    """
    mask_wind = s.data['wind'] > 0.5
    if not np.any(mask_wind):
        return np.nan

    M_shell_g = np.sum(s.data['mass'][mask_wind]) * unit_m   # grams
    r_sh_kpc, _ = find_shock_radius(s)                        # kpc (code-unit length)
    if np.isnan(r_sh_kpc) or r_sh_kpc <= 0:
        return np.nan

    r_sh_cm = r_sh_kpc * unit_l                               # cm
    N_H = (M_shell_g * X_H) / (m_p * 4 * np.pi * r_sh_cm**2)
    return N_H

def column_density_los(s, unit_l):
    sort_idx  = np.argsort(s.data['r'])
    r_sorted  = s.data['r'][sort_idx]
    nH_sorted = s.data['nH_cm'][sort_idx]
    dr_cm     = np.diff(r_sorted, prepend=0) * unit_l   # always ≥ 0 now
    return np.sum(nH_sorted * dr_cm)                    # cm^-2

def column_density_simple(s0, unit_m, unit_l):

    M_shell_g = np.sum(s0.data['mass']) * unit_m   # grams

    boxsize_cm = s0.header['BoxSize'] * unit_l                               # cm
    N_H = (M_shell_g * X_H) / (m_p * 4 * np.pi * boxsize_cm**2)
    return N_H


def cr_energy_over_ewind(s, unit_m, unit_v, t_myr):
    """
    Total CR energy in wind cells divided by total wind energy injected E_w(t).
    """
    mask_wind = np.ones_like(s.data['wind'], dtype=bool)  # s.data['wind'] > 0.5
    if 'cren' not in s.data or not np.any(mask_wind):
        return np.nan

    unit_e  = unit_m * unit_v**2
    E_CR    = np.sum(s.data['cren'][mask_wind] * s.data['mass'][mask_wind]) * unit_e
    E_w     = E_w_dot * (t_myr * myr_to_s)
    E_k = 0.5 * np.sum(s.data['mass'][mask_wind] *s.data['speed'][mask_wind]**2) * unit_e
    if E_w == 0:
        return np.nan
    return E_CR / E_w


# ── collect data ──────────────────────────────────────────────────────────────
print('Loading snapshots ...')
results = {}   # results[label][snap_num] = (t_myr, N_H, ecr_ratio)

for label, cfg in RUNS.items():
    path = cfg['path']
    results[label] = {}
    for snum in SNAP_NUMS:
        snap_file = os.path.join(path, f'snap_{snum:03d}.hdf5')
        if not os.path.exists(snap_file):
            print(f'  {label} snap {snum}: not found, skipping.')
            continue
        try:
            s = load_snap_data(num=snum, snappath=path, snapbase=SNAPBASE)
            s0 = load_snap_data(num=0, snappath=path, snapbase=SNAPBASE)
        except Exception as exc:
            print(f'  {label} snap {snum}: load error — {exc}')
            continue

        unit_v = s.header['UnitVelocity_in_cm_per_s']
        unit_l = s.header['UnitLength_in_cm']
        unit_m = s.header['UnitMass_in_g']

        t_myr = calc_snap_time(s)
        N_H   = column_density_los(s, unit_l)
        ecr_r = cr_energy_over_ewind(s, unit_m, unit_v, t_myr)

        results[label][snum] = (t_myr, N_H, ecr_r)
        print(f'  {label} snap {snum}: t={t_myr:.3f} Myr  N_H={N_H:.2e} cm^-2  E_CR/E_w={ecr_r:.3e}')

# ── plotting ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        14,
    'axes.labelsize':   18,
    'axes.titlesize':   18,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize':  14,
    'ytick.labelsize':  14,
    'xtick.major.size': 7,
    'xtick.minor.size': 4,
    'ytick.major.size': 7,
    'ytick.minor.size': 4,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'axes.linewidth':   1.5,
    'legend.fontsize':  13,
    'legend.title_fontsize': 14,
})

fig, ax = plt.subplots(figsize=(9, 7))

for i,(label, cfg) in enumerate(RUNS.items()):
    color  = cfg['c']
    marker = cfg['marker']
    snaps_data = results.get(label, {})

    # Collect all points for this run to draw a connecting line
    pts = sorted([(d[1], d[2], d[0], sn) for sn, d in snaps_data.items()
                  if not (np.isnan(d[1]) or np.isnan(d[2]))], key=lambda x: x[3])

    if len(pts) > 1:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, lw=1.5, ls='--', alpha=0.5, zorder=1)

    for i, snum in enumerate(SNAP_NUMS):
        if snum not in snaps_data:
            continue
        t_myr, N_H, ecr_r = snaps_data[snum]
        if np.isnan(N_H) or np.isnan(ecr_r):
            continue
        ax.scatter(N_H, ecr_r,
                   color=color, marker=marker,
                   s=SNAP_SIZE[i] * 1.8, alpha=SNAP_ALPHA[i],
                   edgecolors='black', linewidths=0.8,
                   zorder=3, label='_nolegend_')
        h = 'left' if label == r'$\rho = \rho_0 / 10$' else 'right'
        h = 'left' if i == 0 and label == r'$\rho = \rho_0$' else h
        ax.annotate(f'{t_myr:.2f} Myr',
                    xy=(N_H, ecr_r), fontsize=11, fontweight='bold',
                    ha=h, va='bottom',
                    xytext=(5, 5), textcoords='offset points',
                    color=color, alpha=min(SNAP_ALPHA[i] + 0.2, 1.0))

# ── legend ────────────────────────────────────────────────────────────────────
run_handles = [
    Line2D([0], [0], color=cfg['c'], marker=cfg['marker'], lw=0, ms=11,
           mfc=cfg['c'], mec='black', mew=0.8, label=lab)
    for lab, cfg in RUNS.items()
]
size_handles = [
    Line2D([0], [0], color='grey', marker='o', lw=0, ms=np.sqrt(sz) * 0.7,
           alpha=alph, mec='black', mew=0.6,
           label=f'snap {sn}')
    for sn, sz, alph in zip(SNAP_NUMS, SNAP_SIZE, SNAP_ALPHA)
]

# leg1 = ax.legend(handles=run_handles, fontsize=13, framealpha=0.9,
#                  loc='upper left', title='Density', title_fontsize=14,
#                  frameon=True, edgecolor='black')
# ax.add_artist(leg1)
# ax.legend(handles=size_handles, fontsize=13, framealpha=0.9,
#           loc='lower right', title='Snapshot', title_fontsize=14,
#           frameon=True, edgecolor='black')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$N_H \; [\mathrm{cm}^{-2}]$')
ax.set_ylabel(r'$E_\mathrm{CR}/E_\mathrm{w}(t)$')
ax.set_title('Total CR Energy vs. Column Density')
ax.grid(True, alpha=0.25, ls='--', lw=0.8)
ax.tick_params(which='both', top=True, right=True)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'coldens_ecr.png')
fig.savefig(fname, dpi=200, bbox_inches='tight')
print(f'\nPlot saved to: {fname}')
