
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import scienceplots

# plt.style.use(['science'])

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/snap-plotting')
from lib import *

# ── simulation paths ──────────────────────────────────────────────────────────
BASE     = '/cosma8/data/dp317/dc-naza3/homogeneous'
SNAPBASE = 'snap_'

RUNS = {
    # r'$\rho = \rho_0$ no diff': {
    #     'path': BASE + '/output_cool/', 'has_cr': True,
    #     'marker': 'x', 'c': 'C3'
    # },
    # r'$B= 1.855\times 10^{-5} \; \mathrm{G}$+no diff': {
    #     'path': BASE + '/new/output_cbcr/', 'has_cr': True,
    #     'marker': 's', 'c': 'C1'
    # },
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
            # r'$\rho = \rho_0$ 2': {'path': BASE + '/rhov_hires/5/', 'has_cr': True, 'ls': '-', 'c': 'C5','marker': 'o'},

    #     r'$\rho = \rho_0 \times 10$': {'path': BASE + '/old/output_crinc10/', 'has_cr': True, 'ls': '-', 'c': '#1f77b4','marker': 'o'},
    # r'$\rho = \rho_0 / 10$':    {'path': BASE + '/old/output_crred10/', 'has_cr': True, 'ls': '--', 'c': '#ff7f0e','marker': 's'},
    #     r'$\rho = \rho_0$': {'path': BASE + '/old/output_cr/', 'has_cr': True, 'ls': ':', 'c': '#2ca02c','marker': 'D'},

    #                 r'$\rho = \rho_0$':      {'path': BASE + '/rho_vary/5/',    'has_cr': True,  'ls': '--',  'c': 'maroon', 'marker': 's'},
    # # #   r'$\rho = \rho_0 \times 10$':       {'path': BASE + '/rho_vary/50/',    'has_cr': True,  'ls': ':',   'c': 'orange'},
    # r'$\rho = \rho_0 / 10$':      {'path': BASE + '/rho_vary/0.5/',   'has_cr': True,  'ls': '-',   'c': 'teal'},

            r'$n_\mathrm{H}=50$ cm$^{-3}$': {'path': BASE + '/hires/50/',  'ls': '-',  'c': 'C0', 'm': 'o', 'has_cr': True},
    r'$n_\mathrm{H}=5$ cm$^{-3}$':           {'path': BASE + '/hires/5/',   'ls': '--', 'c': 'C1', 'm': 's', 'has_cr': True},
    r'$n_\mathrm{H}=0.5$ cm$^{-3}$':        {'path': BASE + '/hires/0.5/', 'ls': ':',  'c': 'C2', 'm': '^', 'has_cr': True},

}

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rhov2')
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

    data = {k: [] for k in ['time_myr', 'rshell', 'etot_w', 'etot_a', 'ecr_w', 'ecr_a', 'ecr_tot', 'etot_exp', 'ecr_exp', 'mach_w', 'mach_a', 'edis_w', 'edis_a', 'edis_tot']}


    for i in range(1,n_snaps):
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
        ecr_tot = np.sum(mass * s.data['cren']) if has_cr and 'cren' in s.data else 0.0

        mach = s.data['mach']
        mask_a &= mach>2.2
        mask_w &= mach>2.2
        mach_w = np.mean(mach[mask_w])#np.sum(mach[mask_w]*s.data['edis'][mask_w])/np.sum(s.data['edis'][mask_w]) if np.any(mask_w) else 0.0
        mach_a = np.mean(mach[mask_a])#np.sum(mach[mask_a]*s.data['edis'][mask_a])/np.sum(s.data['edis'][mask_a]) if np.any(mask_a) else 0.0

        _, r_shell_u = find_shell_radius(s)

        data['time_myr'].append(calc_snap_time(s))
        data['rshell'].append(r_shell_u)
        data['etot_w'].append(etot_w)
        data['etot_a'].append(etot_a)
        data['ecr_w'].append(ecr_w)
        data['ecr_a'].append(ecr_a)
        data['ecr_tot'].append(ecr_tot)
        data['etot_exp'].append(etot_exp)
        data['ecr_exp'].append(ecr_exp)
        data['mach_w'].append(mach_w)
        data['mach_a'].append(mach_a)
        data['edis_w'].append(np.sum(s.data['edis'][mask_w]))
        data['edis_a'].append(np.sum(s.data['edis'][mask_a]))
        data['edis_tot'].append(np.sum(s.data['edis']))

    snap_energy[label] = {k: np.array(v) for k, v in data.items()}
    print(f'  {label}: {len(data["time_myr"])} snaps loaded.')

# ── Plotting ─────────────────────────────────────────────────────────────
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

USE_TIME = True # True → x-axis is time [Myr]; False → x-axis is R_sh [kpc]

fig, axes = plt.subplots(3, 2, figsize=(12, 16))
ax_tot, ax_cr, ax_mach, ax_crtot, ax_edis, ax_edistot = axes.ravel()

PHASE_STYLE = {
    'SW':  {'ls': '-',  'fill': 'full'},
    'SAM': {'ls': '--', 'fill': 'none'},
}

# Get reference units from an actual run for accurate E_w scaling
s0 = load_snap_data(num=0, snappath=list(RUNS.values())[0]['path'], snapbase=SNAPBASE)
unit_v = s0.header['UnitVelocity_in_cm_per_s']
unit_l = s0.header['UnitLength_in_cm']
unit_m = s0.header['UnitMass_in_g']
unit_e = unit_m * unit_v**2
unit_t = unit_l / unit_v

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

    x    = time          if USE_TIME else se['rshell']
    name = label.replace('\n', ' ')

    # Total Energy Panel
    ax_tot.plot(
        x, se['etot_w'] / E_w,
        ls=PHASE_STYLE['SW']['ls'], marker=marker, color=color,
        ms=6, lw=2.0, mfc=color, mec='black', mew=0.5
    )
    ax_tot.plot(
        x, se['etot_a'] / E_w,
        ls=PHASE_STYLE['SAM']['ls'], marker=marker, color=color,
        ms=6, lw=2.0, mfc='white', mec=color, mew=1.0
    )

    # CR Energy by phase Panel
    if RUNS[label]['has_cr']:
        ax_cr.plot(
            x, se['ecr_w'] / E_w,
            ls=PHASE_STYLE['SW']['ls'], marker=marker, color=color,
            ms=6, lw=2.0, mfc=color, mec='black', mew=0.5
        )
        ax_cr.plot(
            x, se['ecr_a'] / E_w,
            ls=PHASE_STYLE['SAM']['ls'], marker=marker, color=color,
            ms=6, lw=2.0, mfc='white', mec=color, mew=1.0
        )
        # ax_cr.plot(
        #     x, se['ecr_exp'] / E_w,
        #     ls=':', marker=marker, color=color,
        #     ms=6, lw=2.0, mfc='white', mec=color, mew=1.0
        # )

    # Mach number by phase Panel
    ax_mach.plot(
        x, se['mach_w'],
        ls=PHASE_STYLE['SW']['ls'], marker=marker, color=color,
        ms=6, lw=2.0, mfc=color, mec='black', mew=0.5
    )
    ax_mach.plot(
        x, se['mach_a'],
        ls=PHASE_STYLE['SAM']['ls'], marker=marker, color=color,
        ms=6, lw=2.0, mfc='white', mec=color, mew=1.0
    )

    # Total CR Energy Panel (panel b style from crenergy.py)
    if RUNS[label]['has_cr'] and np.any(se['ecr_tot'] != 0):
        ax_crtot.plot(
            x, se['ecr_tot']/ E_w,
            ls='-', marker=marker, color=color,
            ms=6, lw=2.0, mfc=color, mec='black', mew=0.5, label=name
        )

    # Energy dissipation rate per phase (normalised by L_AGN)
    edis_w_rate = se['edis_w'] * unit_e / unit_t / L_AGN
    edis_a_rate = se['edis_a'] * unit_e / unit_t / L_AGN
    ax_edis.plot(
        x, edis_w_rate,
        ls=PHASE_STYLE['SW']['ls'], marker=marker, color=color,
        ms=6, lw=2.0, mfc=color, mec='black', mew=0.5
    )
    ax_edis.plot(
        x, edis_a_rate,
        ls=PHASE_STYLE['SAM']['ls'], marker=marker, color=color,
        ms=6, lw=2.0, mfc='white', mec=color, mew=1.0
    )

    # Total energy dissipation rate (all particles, normalised by L_AGN)
    edis_tot_rate = se['edis_tot'] * unit_e / unit_t / L_AGN
    ax_edistot.plot(
        x, edis_tot_rate,
        ls='-', marker=marker, color=color,
        ms=6, lw=2.0, mfc=color, mec='black', mew=0.5, label=name
    )

_xlabel = r'$t$ [Myr]' if USE_TIME else r'$R_\mathrm{sh}$ [kpc]'

ax_tot.set(xlabel=_xlabel, ylabel=r'$E_\mathrm{total} / E_\mathrm{w}$',   title='Total Energy by Phase')
ax_cr.set( xlabel=_xlabel, ylabel=r'$E_\mathrm{CR} / E_\mathrm{w}$',      title='CR Energy by Phase')
ax_mach.set(xlabel=_xlabel, ylabel=r'$\mathcal{M}$',                        title=' Mach Number by Phase')
ax_crtot.set(xlabel=_xlabel, ylabel=r'$E_\mathrm{CR} / E_\mathrm{wind}$',  title='Total CR Energy',
             xscale='log', yscale='log')
ax_edis.set(xlabel=_xlabel, ylabel=r'$\dot{E}_\mathrm{dis} / L_\mathrm{AGN}$',
            title='Energy Dissipation Rate by Phase')
ax_edistot.set(xlabel=_xlabel, ylabel=r'$\dot{E}_\mathrm{dis} / L_\mathrm{AGN}$',
               title='Total Energy Dissipation Rate')

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

for ax in (ax_tot, ax_cr, ax_mach):
    leg_runs = ax.legend(handles=run_handles+phase_handles, fontsize=8, framealpha=0.7, loc='best')
    ax.add_artist(leg_runs)
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_yscale("log")
    ax.set_xscale("log")

# ax_crtot.legend(fontsize=8, framealpha=0.7, loc='best')
ax_crtot.grid(True, alpha=0.3, ls='--')


ax_edis.legend(handles=run_handles+phase_handles, fontsize=8, framealpha=0.7, loc='best')
ax_edis.set_yscale("log")
ax_edis.set_xscale("log")
ax_edis.grid(True, alpha=0.3, ls='--')

# ax_edistot.legend(fontsize=8, framealpha=0.7, loc='best')
ax_edistot.set_yscale("log")
ax_edistot.set_xscale("log")
ax_edistot.grid(True, alpha=0.3, ls='--')

_plain_fmt = FuncFormatter(lambda x, _: f'{x:.10g}')
for ax in [ax_tot, ax_cr, ax_mach, ax_crtot, ax_edis, ax_edistot]:
    ax.xaxis.set_major_formatter(_plain_fmt)
    ax.yaxis.set_major_formatter(_plain_fmt)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'phase_energy_plots2.png')
fig.savefig(fname, dpi=150)
print(f'\n✓ Plot saved to: {fname}')
