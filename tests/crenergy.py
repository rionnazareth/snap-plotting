#!/cosma/apps/dp317/dc-naza3/renv/bin/python
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots

# plt.style.use(['science'])

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/snap-plotting')
from lib import *
from matplotlib.colors import LogNorm

# ── simulation paths ──────────────────────────────────────────────────────────
BASE     = '/cosma8/data/dp317/dc-naza3/homogeneous'
SNAPBASE = 'snap_'

RUNS = {
    # r'$\rho = \rho_0$': {'path': BASE + '/old/output_cr/', 'has_cr': True},
    # r'$n_\mathrm{H,0} \sim 5 cm^{-3}$': {'path': BASE + '/old/output_cr600/', 'has_cr': True},
    #     r'$n_\mathrm{H,0} \sim 5 cm^{-3}$+CR diff.': {'path': BASE + '/old/output_cr/', 'has_cr': True},
    #         r'$n_\mathrm{H,0} \sim 5 cm^{-3}$+CR diff.+$B_x=10^{-6} \; \mathrm{G}$': {'path': BASE + '/output_uni/', 'has_cr': True},
    # r'no cooling': {
    #     'path': BASE + '/old/output_cr600/', 'has_cr': True,
    #     'marker': 'o', 'color': 'C0'
    # },
    # r'$B= 0$+no diff': {
    #     'path': BASE + '/old/output_cr600/', 'has_cr': True,
    #     'marker': 's', 'color': 'C1'
    # },
    # r'cooling': {
    #     'path': BASE + '/output_cool/', 'has_cr': True,
    #     'marker': 'D', 'color': 'C2'
    # },
    # r'no CRs': {
    #     'path': BASE + '/old/output_homo/', 'has_cr': True,
    #     'marker': 'X', 'color': 'C3'
    # },

    #     r'Hydro only':      {'path': BASE + '/new/output_cnocr/',    'has_cr': False,  'ls': '--',  'c': 'maroon', },
    #   r'Hydro+B fields':       {'path': BASE + '/new/output_cbf/',    'has_cr': True,  'ls': ':',   'c': 'orange'},
    # r'Hydro+B fields+CRs':      {'path': BASE + '/new/output_cbcr/',   'has_cr': True,  'ls': '-',   'c': 'orange'},

    #         r'$\rho = \rho_0$':      {'path': BASE + '/rho_vary/5/',    'has_cr': True,  'ls': '--',  'c': 'maroon', },
    # #   r'$\rho = \rho_0 \times 10$':       {'path': BASE + '/rho_vary/50/',    'has_cr': True,  'ls': ':',   'c': 'orange'},
    # r'$\rho = \rho_0 / 10$':      {'path': BASE + '/rho_vary/0.5/',   'has_cr': True,  'ls': '-',   'c': 'teal'},
    # r'with diffusion':      {'path': BASE + '/old/output_cr/',   'has_cr': True,  'ls': '-',   'c': 'green'},

        r'$\rho = \rho_0 \times 10$': {'path': BASE + '/rho_vary/50/', 'has_cr': True, 'ls': '-', 'c': 'C0','marker': 'o'},
    r'$\rho = \rho_0 / 10$':    {'path': BASE + '/rho_vary/0.5/', 'has_cr': True, 'ls': '--', 'c': 'C1','marker': 's'},
        r'$\rho = \rho_0$': {'path': BASE + '/rho_vary/5/', 'has_cr': True, 'ls': ':', 'c': 'C2','marker': 'D'},

    # r'$\rho = \rho_0 \times 10$': {'path': BASE + '/old/output_crinc10/', 'has_cr': True},
    # r'$\rho = \rho_0 / 10$':    {'path': BASE + '/old/output_crred10/', 'has_cr': True},
    # r'no cooling': {'path': BASE + '/output_uni/', 'has_cr': True},
    # r'cooling': {'path': BASE + '/output_cdiff/', 'has_cr': True},
    # r'no cooling no B fields and diff': {'path': BASE + '/old/output_cr600/', 'has_cr': True},
    # r'cooling no B fields and diff': {'path': BASE + '/output_cool/', 'has_cr': True},
    # r'CR cooling no B fields': {'path': BASE + '/output_cr/', 'has_cr': True},
    # r'$\rho = \rho_0$+B = 1e-6 G': {'path': BASE + '/output_uni/', 'has_cr': True},
}

COLORS = {
        # r'$n_\mathrm{H,0} \sim 5 cm^{-3}$': 'purple',
        # r'$n_\mathrm{H,0} \sim 5 cm^{-3}$+CR diff.': 'steelblue',
        # r'$n_\mathrm{H,0} \sim 5 cm^{-3}$+CR diff.+$B_x=10^{-6} \; \mathrm{G}$': 'darkgreen',
    # r'$\rho = \rho_0$': 'steelblue',
    # r'$\rho = \rho_0 \times 10$': 'darkorange',
    # r'$\rho = \rho_0 / 10$':    'purple',
    # r'$\rho = \rho_0$+B = 1e-6 G': 'cyan',
    # r'no cooling': 'green',
    # r'cooling': 'red',
    # r'no cooling no B fields and diff': 'green',
    # r'cooling no B fields and diff': 'red',
    # r'CR cooling no B fields': 'purple'
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
value = 'mach'
value2 = 'pres'

for label, cfg in RUNS.items():
    path = cfg['path']
    has_cr = cfg['has_cr']
    n_snaps = _count_snaps(path)
    
    if n_snaps == 0:
        print(f'  {label}: no snapshots found.')
        continue

    data = {k: [] for k in ['time_myr', 'ekin', 'etherm', 'ecr', 'ebfld', 'epot', 'etotal', 'fshock', 'rshock', 'rshell',value, value2, 'edis','max_mach']}

    for i in range(n_snaps):
        try:
            s = load_snap_data(num=i, snappath=path, snapbase=SNAPBASE)
        except Exception as exc:
            print(f"Could not load snap {i} for {label}: {exc}")
            continue

        unit_v = s.header['UnitVelocity_in_cm_per_s']
        unit_l = s.header['UnitLength_in_cm'] 
        unit_m = s.header['UnitMass_in_g']
        unit_t = unit_l / unit_v
        unit_rho = unit_m / unit_l**3
            
        mass = s.data['mass']
        vel  = s.data['vel']
        u_   = s.data['u']

        ek = 0.5 * np.sum(mass * np.sum(vel**2, axis=1))
        et = np.sum(mass * u_)
        
        ec = 0.0
        if has_cr and 'cren' in s.data: 
            ec = np.sum(mass*s.data['cren'])
            
        eb = 0.0
        if 'bflden' in s.data:
            eb = np.sum(mass * s.data['bflden'])
        eb=0.0

        ep = 0.0 

        ev = 0.0
        try:
            mach = s.data['mach']
            ev = np.sum(mach*s.data['edis'])/np.sum(s.data['edis'])
        except KeyError:
            pass

        edis_tot = 0.0
        if 'edis' in s.data:
            edis_tot = np.sum(s.data['edis'])
        
        # if 'mach' in s.data:
        #     rad_wind = 0.0078125
        #     mask = np.ones_like(s.data['r'], dtype=bool)
        #     # mask =  (s.data['r'] >= rad_wind) 
        #     # mask = (s.data['wind'] >= 0.5) & (s.data['vrad']*unit_v/1e5 > 10)
        #     mask = (s.data['mach']>3)
        #     mach = s.data['mach']
        #     max_mach = (mach[mask]).max() #if np.any(mask) else 0.0
        

        fshock, rshock = find_shock_radius(s, r_range=(1e-3,1))
        r_shell_l, r_shell_u = find_shell_radius(s)

        data['time_myr'].append(calc_snap_time(s))
        data['ekin'].append(ek)
        data['etherm'].append(et)
        data['ecr'].append(ec)
        data['ebfld'].append(eb)
        data['epot'].append(-ep)
        data['etotal'].append(ek + et + ec - ep + eb)

        data['fshock'].append(fshock)
        data['rshock'].append(rshock)
        data['rshell'].append(r_shell_u)
        data[value].append(ev)
        data[value2].append(np.sum(s.data[value2]))
        data['edis'].append(edis_tot)
        # data['max_mach'].append(max_mach)
    snap_energy[label] = {k: np.array(v) for k, v in data.items()}
    print(f'  {label}: {len(data["time_myr"])} snaps loaded.')

# ── Plotting ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
ax_norm, ax_cr, ax_parts, ax_value = axes

unit_v = s.header['UnitVelocity_in_cm_per_s']
unit_l = s.header['UnitLength_in_cm'] 
unit_m = s.header['UnitMass_in_g']
unit_t = unit_l / unit_v
unit_rho = unit_m / unit_l**3
unit_e = unit_m * unit_v**2

c = 3e10 # Speed of light in cm/s

# Time conversion factor (Myr to seconds)
myr_to_s = 1e6 * 365.25 * 24 * 3600

L_AGN = 1e45  # erg/s
v_w = 1e4 * 1e5 # Wind velocity in cm/s
beta = v_w/c
E_w_dot = 0.5 * beta * L_AGN

rho_fac = np.sqrt(np.array([1, 10, 0.1]))
rho0 = 0.0148  # Initial density in code units (from initial conditions)
t_free = calculate_R_free(beta = beta, tau = 1, b = 1, L_AGN = L_AGN, rho_0 = rho0*unit_rho)[1]/(3.15e13)#value of free time in Myears




for label, se in snap_energy.items():
    try:
        col = RUNS[label].get('c', 'C0')
    except KeyError:
        col = RUNS[label].get('color', 'C0')
    time = se['time_myr']
    E_w = E_w_dot /unit_e * time * myr_to_s

    time = time  /t_free #* rho_fac[list(RUNS.keys()).index(label)]# Scale time by sqrt(rho) to align features
    name = label.replace('\n', ' ')
    
    # if list(RUNS.keys()).index(label) == 0:
    E0_tot = se['etotal'][0]
    
    # (a) Total Energy
    ax_norm.plot(se['rshell'], (se['etotal']-E0_tot)/E_w, 'o-', color=col, ms=4, label=name)
    
    # (b) CR Energy
    if RUNS[label]['has_cr'] and np.any(se['ecr'] != 0):
        E0_cr = se['ecr'][0] if se['ecr'][0] != 0 else 1
        # 
        markers = ['o', 's', '^', 'D', 'v', 'p', '*']
        sym = markers[list(RUNS.keys()).index(label)]
        
        # Plot edis, for instance, vs time or rshell. Let's assume edis is the Y-axis, and coloring by time.
        # OR scatter(rshell, edis, colored by time). Wait, the easiest is to scatter and let edis be mapped to color.
        # "plot a colorbar and plot s.data['edis']"
        
        # Actually maybe they want edis on the Y-axis and NO ecr/etotal.
        # Add a proxy artist for the legend to show the symbol, since scatter with a colormap often loses the legend handle
        ax_cr.plot(se['rshell'], se['ecr']/E_w , '--', ms=4,color=col, label=name)
        # sc = ax_cr.scatter(se['rshell'], se['ecr'] / E_w, c=se['edis'] / E_w, marker=sym, cmap='viridis')
        sc = ax_cr.scatter(se['rshell'], se['ecr'] / E_w, c=col, marker=sym, cmap='viridis')#se['edis']*(unit_e/unit_t)/L_AGN
        # if not hasattr(ax_cr, 'colorbar_added'):
        #     cvals = se['edis'] * (unit_e / unit_t) / L_AGN
        #     cvals = cvals[cvals > 0]
        #     sc.set_norm(LogNorm())#vmin=0.0181, vmax=0.0186
        #     cbar = fig.colorbar(sc, ax=ax_cr)
        #     cbar.set_label(r'$\dot{E}_\mathrm{dis} / L_\mathrm{AGN}$', rotation=270, labelpad=15)
        #     # cbar.set_label(r'$\mathcal{M}_\mathrm{SW}$', rotation=270, labelpad=15)
        #     ax_cr.colorbar_added = True

        pass
        # ax_cr.set_ylim(0.10, 0.26)  # Set y-limits for better visibility

    # (c) Components Breakdown
    E0_abs = np.abs(E0_tot)
    ax_parts.plot(se['rshell'], se['ekin'] / E0_abs,   's-', color=col, ms=3, alpha=0.6, label=f'{name} $E_\\mathrm{{kin}}$')
    ax_parts.plot(se['rshell'], se['etherm'] / E0_abs, '^-', color=col, ms=3, alpha=0.6, label=f'{name} $E_\\mathrm{{th}}$')
    ax_parts.plot(se['rshell'], se['ecr'] / E0_abs,    'v--', color=col, ms=3, alpha=0.6, label=f'{name} $E_\\mathrm{{cr}}$')

    # (d) Custom Value
    if np.any(se[value] != 0):
        if list(RUNS.keys()).index(label) == 0:
            E0_value = se[value2][0] 
        ax_value.plot(se['rshell'], se[value], 'o-', color=col, ms=4, label=name)

ax_norm.set(xlabel=r'$R_\mathrm{sh}$', ylabel=r'$E_\mathrm{total}(t) / E_\mathrm{wind}$', title='(a) Total energy normalised')
ax_norm.axhline(1, color='k', ls='--', alpha=0.5)

ax_cr.set(xlabel=r'$R_\mathrm{sh}$ [kpc]', ylabel=r'$E_\mathrm{CR} / E_\mathrm{wind} $', title='(b) Dissipated Energy', xscale='log', yscale='log')

ax_parts.set(xlabel=r'$R_\mathrm{sh}$', ylabel=r'$E_\mathrm{CR}/ |E_\mathrm{total,0}|$', title='(c) Energy components')

ax_value.set(xlabel=r'$R_\mathrm{sh}$', ylabel=f'$\mathcal{{M}}$', title='(d)  Custom value evolution')

for ax in axes:
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(left=0)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'simple_energy_plots.png')
fig.savefig(fname, dpi=150)
print(f'\n✓ Plot saved to: {fname}')
