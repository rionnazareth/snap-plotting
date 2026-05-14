#!/cosma/apps/dp317/dc-naza3/renv/bin/python
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots

plt.style.use(['science'])

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/snap-plotting')
from lib import *

# ── simulation paths ──────────────────────────────────────────────────────────
BASE     = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion'
SNAPBASE = 'snap_'

RUNS = {
    r'$\rho = \rho_0$ no diff': {
        'path': BASE + '/output_cool/', 'has_cr': True,
        'marker': 'x', 'c': 'C3'
    },
    r'$B= 1.855\times 10^{-5} \; \mathrm{G}$+no diff': {
        'path': BASE + '/new/output_cbcr/', 'has_cr': True,
        'marker': 's', 'c': 'C1'
    },
    # r'cooling': {
    #     'path': BASE + '/output_cool/', 'has_cr': True,
    #     'marker': 'D', 'c': 'C2'
    # },
    # r'no CRs': {
    #     'path': BASE + '/old/output_homo/', 'has_cr': True,
    #     'marker': 'X', 'c': 'C3'
    # },
    # r'$B_{\phi} = 10^{-6} \; \mathrm{G}$': {
    #     'path': BASE + '/output_azi/', 'has_cr': True,
    #     'marker': 'P', 'c': 'C4'
    # },

    #         r'Hydro only':      {'path': BASE + '/new/output_cnocr/',    'has_cr': False,  'ls': '--',  'c': 'maroon', 'marker': 'o'},
    #   r'Hydro+B fields':       {'path': BASE + '/new/output_cbf/',    'has_cr': True,  'ls': ':',   'c': 'orange', 'marker': 's'},
    # r'Hydro+B fields+CRs':      {'path': BASE + '/new/output_cbcr/',   'has_cr': True,  'ls': '-',   'c': 'teal', 'marker': 'D'},

    #     r'$\rho = \rho_0 \times 10$': {'path': BASE + '/old/output_crinc10/', 'has_cr': True, 'ls': '-', 'c': 'C0','marker': 'o'},
    # r'$\rho = \rho_0 / 10$':    {'path': BASE + '/old/output_crred10/', 'has_cr': True, 'ls': '--', 'c': 'C1','marker': 's'},
    #     r'$\rho = \rho_0$': {'path': BASE + '/old/output_cr/', 'has_cr': True, 'ls': ':', 'c': 'C2','marker': 'D'},

}

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bfield_amp')
os.makedirs(OUTDIR, exist_ok=True)

def _count_snaps(path):
    n = 0
    while os.path.exists(os.path.join(path, f'snap_{n:03d}.hdf5')):
        n += 1
    return n

print('Reading snapshots ...')
snap_energy = {}

for label, cfg in RUNS.items():
    path = cfg['path']
    has_cr = cfg['has_cr']
    n_snaps = _count_snaps(path)
    
    if n_snaps == 0:
        print(f'  {label}: no snapshots found.')
        continue

    data = {k: [] for k in ['time_myr', 'rshell', 'etot_w', 'etot_a', 'ecr_w', 'ecr_a', 'etot_exp', 'ecr_exp']}

    for i in range(n_snaps):
        try:
            s = load_snap_data(num=i, snappath=path, snapbase=SNAPBASE)
        except Exception as exc:
            continue
            
        mass = s.data['mass']
        vel  = s.data['vel']
        u_   = s.data['u']
        wind = s.data['wind']

        unit_v = s.header['UnitVelocity_in_cm_per_s']
        unit_m = s.header['UnitMass_in_g']
        
        # Phases mapping
        mask_w = wind >= 0.5
        mask_a = (wind < 0.5) & (s.data['vrad']*unit_v/1e5 > 10)  # Arbitrary threshold to define SAM
        r_sh, r_rev = find_shock_radius(s)
        mask_exp = s.data['r'] > r_sh

        # Compute energy by phase mask
        def calc_phase_energies(m):
            ek = 0.5 * np.sum(mass[m] * np.sum(vel[m]**2, axis=1))
            et = np.sum(mass[m] * u_[m])
            ec = np.sum(mass[m] * s.data['cren'][m]) if has_cr and 'cren' in s.data else 0.0
            return ek + et + ec, ec
            
        etot_w, ecr_w = calc_phase_energies(mask_w)
        etot_a, ecr_a = calc_phase_energies(mask_a)
        etot_exp, ecr_exp = calc_phase_energies(mask_exp)

        _, r_shell_u = find_shell_radius(s)

        data['time_myr'].append(calc_snap_time(s))
        data['rshell'].append(r_shell_u)
        data['etot_w'].append(etot_w)
        data['etot_a'].append(etot_a)
        data['ecr_w'].append(ecr_w)
        data['ecr_a'].append(ecr_a)
        data['etot_exp'].append(etot_exp)
        data['ecr_exp'].append(ecr_exp)

    snap_energy[label] = {k: np.array(v) for k, v in data.items()}
    print(f'  {label}: {len(data["time_myr"])} snaps loaded.')

# ── Plotting ─────────────────────────────────────────────────────────────
fig, (ax_tot, ax_cr) = plt.subplots(1, 2, figsize=(12, 5))

PHASE_STYLE = {
    'SW':  {'ls': '-',  'fill': 'full'},
    'SAM': {'ls': '--', 'fill': 'none'},
}

# Get reference units from an actual run for accurate E_w scaling
s0 = load_snap_data(num=0, snappath=list(RUNS.values())[0]['path'], snapbase=SNAPBASE)
unit_v = s0.header['UnitVelocity_in_cm_per_s']
unit_m = s0.header['UnitMass_in_g']
unit_e = unit_m * unit_v**2

c = 3e10
myr_to_s = 1e6 * 365.25 * 24 * 3600
L_AGN = 1e45  
v_w = 1e4 * 1e5 
beta = v_w / c
E_w_dot = 0.5 * beta * L_AGN

for label, se in snap_energy.items():
    cfg = RUNS.get(label, {})
    marker = cfg.get('marker', 'o')
    try:
        color = RUNS[label].get('c')
    except KeyError:
        color = RUNS[label].get('color')
    time = se['time_myr']
    E_w = (E_w_dot / unit_e) * time * myr_to_s
    # Small offset to avoid dividing by perfectly zero 
    E_w = np.where(E_w == 0, 1e-10, E_w) 
    
    r_sh = se['rshell']
    name = label.replace('\n', ' ')
    
    # Total Energy Panel
    ax_tot.plot(
        r_sh, se['etot_w'] / E_w,
        ls=PHASE_STYLE['SW']['ls'], marker=marker, color=color,
        ms=6, lw=2.0, mfc=color, mec='black', mew=0.5
    )
    ax_tot.plot(
        r_sh, se['etot_a'] / E_w,
        ls=PHASE_STYLE['SAM']['ls'], marker=marker, color=color,
        ms=6, lw=2.0, mfc='white', mec=color, mew=1.0
    )

    # CR Energy Panel
    if RUNS[label]['has_cr']:
        ax_cr.plot(
            r_sh, se['ecr_w'] / E_w,
            ls=PHASE_STYLE['SW']['ls'], marker=marker, color=color,
            ms=6, lw=2.0, mfc=color, mec='black', mew=0.5
        )
        ax_cr.plot(
            r_sh, se['ecr_a'] / E_w,
            ls=PHASE_STYLE['SAM']['ls'], marker=marker, color=color,
            ms=6, lw=2.0, mfc='white', mec=color, mew=1.0
        )
        # ax_cr.plot(
        #     r_sh, se['ecr_exp'] / E_w,
        #     ls=':', marker=marker, color=color,
        #     ms=6, lw=2.0, mfc='white', mec=color, mew=1.0
        # )

ax_tot.set(xlabel=r'$R_\mathrm{sh}$ [kpc]', ylabel=r'$E_\mathrm{total} / E_\mathrm{w}$', title='Total Energy by Phase')
ax_cr.set(xlabel=r'$R_\mathrm{sh}$ [kpc]', ylabel=r'$E_\mathrm{CR} / E_\mathrm{w}$', title='CR Energy by Phase')

# Build two clean legends: one for run, one for phase
run_handles = [
    Line2D([0], [0],
           color=RUNS.get(k, {}).get('c', 'C0'),
           marker=RUNS.get(k, {}).get('marker', 'o'),
           lw=0, ms=7,
           mfc=RUNS.get(k, {}).get('c', 'C0'),
           mec='black', mew=0.5, label=k)
    for k in RUNS.keys()
]
phase_handles = [
    Line2D([0], [0], color='black', ls='-',  lw=2, marker='o', ms=6, mfc='black', mec='black', label='SW'),
    Line2D([0], [0], color='black', ls='--', lw=2, marker='o', ms=6, mfc='white', mec='black', label='SAM'),
]

for ax in (ax_tot, ax_cr):
    leg_runs = ax.legend(handles=run_handles+phase_handles, fontsize=8, framealpha=0.7, loc='best')
    ax.add_artist(leg_runs)
    # ax.legend(handles=phase_handles, fontsize=8, framealpha=0.7, loc='best')
    ax.grid(True, alpha=0.3, ls='--')
    # ax.set_xlim(left=0)
    ax.set_yscale("log") # Standard to view phase-split normalization over time/radius as log
    ax.set_xscale("log") # Shell radius often spans orders of magnitude, so log scale can help visualize trends across all scales

plt.tight_layout()
fname = os.path.join(OUTDIR, 'phase_energy_plots.png')
fig.savefig(fname, dpi=150)
print(f'\n✓ Plot saved to: {fname}')
