#!/cosma/apps/dp317/dc-naza3/renv/bin/python
"""
energy_conservation.py
======================
Checks energy conservation across three simulations and examines how
energy is partitioned between thermal gas and cosmic rays (CRs).

Simulations
-----------
  output2/      – no CRs (purely thermal hydrodynamics)
  output_cr/    – with CRs, no diffsion
  output_crexps/– with CRs and CR diffusion (exponential spectrum)

Plots produced
--------------
  1. energy_total.png          – total energy (ekin + etherm [+ ecr]) vs time
                                 normalised to E(t=0) for each run
  2. energy_deviation.png      – fractional deviation ΔE/E₀ = (E(t)−E₀)/E₀
  3. energy_components.png     – stacked kinetic / thermal / CR energies vs time
  4. cr_fraction.png           – f_CR = E_CR / E_total vs time  (CR runs only)
  5. xcr_radial.png            – X_cr = P_cr / P_th radial profile at last
                                 available snapshot (CR runs only)
  6. cr_energy_budget.png      – CR energy budget: injected vs cooling losses vs stored
  7. cr_summary.png            – Combined summary: X_cr and f_CR on one figure
  8. energy_crosssim.png      – Cross-simulation check: E_mech (no CR) vs E_mech+CR
  9. energy_from_snapshots.png – Total energy computed directly from snapshots

All figures are saved to tests/plots/.
"""

import sys, os

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/arepo-snap-util')
sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/gasCloudNfw/snap-plotting')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import arepo_run as arun
import astropy.units as u

# ── constants ─────────────────────────────────────────────────────────────────
GAMMA    = 5. / 3.
GAMMA_CR = 4. / 3.
k_B      = 1.381e-16    # erg K⁻¹
m_p      = 1.66e-24     # g
unit_v   = 1.e5         # cm s⁻¹   (1 km/s)

# ── simulation paths ──────────────────────────────────────────────────────────
BASE     = '/cosma8/data/dp317/dc-naza3/gasCloudNfw'
SNAPBASE = 'snap_'

RUNS = {
    'no-CR\n(output2)':         {'path': BASE + '/output2/',     'has_cr': False},
    'CR (no diff)\n(output_cr)':  {'path': BASE + '/output_cr/',   'has_cr': True},
    'CR (diff)\n(output_crexps)': {'path': BASE + '/output_crexps/', 'has_cr': True},
}

COLORS = {
    'no-CR\n(output2)':          'dimgrey',
    'CR (no diff)\n(output_cr)':  'steelblue',
    'CR (diff)\n(output_crexps)': 'darkorange',
}

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(OUTDIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: unit conversion – code time → Myr
# ─────────────────────────────────────────────────────────────────────────────
_UNIT_L = 3.08568e21   # cm  (1 kpc)
_UNIT_V = 1.00e5       # cm/s (1 km/s)
_UNIT_T = _UNIT_L / _UNIT_V  # s


def code_time_to_Myr(t_code):
    """Convert code-unit time to Myr."""
    t_s = t_code * _UNIT_T
    return (t_s * u.s).to(u.Myr).value


# ─────────────────────────────────────────────────────────────────────────────
# Energy.txt reader
# Arepo energy.txt column layout (0-indexed):
#   0  : time
#   1  : E_kin  (total, all types)
#   2  : E_pot  (total, all types)
#   3  : E_therm(total, all types = internal energy × mass)
#   4-6: same for particle type 0 (gas)
#   7-9: type 1 ... 19-21: type 5
#   22 : total mass
#   23-28: per-type mass / momentum
# ─────────────────────────────────────────────────────────────────────────────
def read_energy(run_path):
    """
    Returns dict with numpy arrays:
        time, ekin, epot, etherm, etotal
    All energies in code units.
    """
    fpath = os.path.join(run_path, 'energy.txt')
    if not os.path.exists(fpath):
        raise FileNotFoundError(f'energy.txt not found in {run_path}')
    data = np.loadtxt(fpath)
    return {
        'time':   data[:, 0],
        'ekin':   data[:, 1],
        'epot':   data[:, 2],
        'etherm': data[:, 3],
        # total mechanical + thermal (no CR yet)
        'emech':  data[:, 1] + data[:, 2] + data[:, 3],
    }


def read_crenergy(run_path):
    """
    Returns dict with numpy arrays (from crenergy.txt):
        time, ecr, ecr_injected, ecr_hadronic, ecr_coulomb, ecr_total_cool
    """
    fpath = os.path.join(run_path, 'crenergy.txt')
    if not os.path.exists(fpath):
        return None
    data = np.loadtxt(fpath)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return {
        'time':        data[:, 0],
        'ecr':         data[:, 1],
        'ecr_inject':  data[:, 2],
        'ecr_hadronic':data[:, 3],
        'ecr_coulomb': data[:, 4],
        'ecr_total_cool': data[:, 5],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load energy data for all runs
# ─────────────────────────────────────────────────────────────────────────────
print('Reading energy files …')
energy  = {}
cre     = {}

for label, cfg in RUNS.items():
    energy[label] = read_energy(cfg['path'])
    # Convert time to Myr
    energy[label]['time_myr'] = code_time_to_Myr(energy[label]['time'])
    if cfg['has_cr']:
        cre[label] = read_crenergy(cfg['path'])
        if cre[label] is not None:
            cre[label]['time_myr'] = code_time_to_Myr(cre[label]['time'])
            # CR energy shares the same time grid as energy.txt
            energy[label]['ecr']    = cre[label]['ecr']
            energy[label]['etotal'] = energy[label]['emech'] + cre[label]['ecr']
        else:
            energy[label]['ecr']    = np.zeros_like(energy[label]['time'])
            energy[label]['etotal'] = energy[label]['emech']
    else:
        energy[label]['ecr']    = np.zeros_like(energy[label]['time'])
        energy[label]['etotal'] = energy[label]['emech']

    print(f"  {label.replace(chr(10),' '):<35s}: {len(energy[label]['time'])} time steps, "
          f"E₀ = {energy[label]['etotal'][0]:.4g} [code]")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 – Total energy vs time (normalised to E₀)
# ─────────────────────────────────────────────────────────────────────────────
print('\nPlot 1: normalised total energy vs time …')

fig, ax = plt.subplots(figsize=(8, 5))

for label, e in energy.items():
    t   = e['time_myr']
    E   = e['etotal']
    E0  = E[0]
    ax.plot(t, E / E0, color=COLORS[label], lw=2,
            label=label.replace('\n', ' '))

ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.5, label='perfect conservation')
ax.set_xlabel('Time [Myr]', fontsize=12)
ax.set_ylabel(r'$E_\mathrm{total}(t)\;/\;E_\mathrm{total}(0)$', fontsize=12)
ax.set_title('Total energy normalised to initial value', fontsize=13)
ax.legend(fontsize=9, framealpha=0.7)
ax.grid(True, alpha=0.3, ls='--')
ax.set_xlim(left=0)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'energy_total.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 – Fractional energy deviation ΔE/E₀
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 2: fractional energy deviation …')

fig, ax = plt.subplots(figsize=(8, 5))

for label, e in energy.items():
    t   = e['time_myr']
    E   = e['etotal']
    E0  = E[0]
    dE  = (E - E0) / E0
    ax.plot(t, dE * 100., color=COLORS[label], lw=2,
            label=label.replace('\n', ' '))

ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Time [Myr]', fontsize=12)
ax.set_ylabel(r'$\Delta E / E_0\;\;[\%]$', fontsize=12)
ax.set_title(r'Fractional energy deviation  $\Delta E/E_0 = (E(t)-E_0)/E_0$', fontsize=13)
ax.legend(fontsize=9, framealpha=0.7)
ax.grid(True, alpha=0.3, ls='--')
ax.set_xlim(left=0)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f%%'))

plt.tight_layout()
fname = os.path.join(OUTDIR, 'energy_deviation.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 – Energy components vs time (one panel per run)
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 3: energy components …')

fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=False)

for ax, (label, e) in zip(axes, energy.items()):
    t    = e['time_myr']
    E0   = e['etotal'][0]

    ax.plot(t, e['ekin']   / E0, color='royalblue',  lw=2, label=r'$E_\mathrm{kin}$')
    ax.plot(t, e['etherm'] / E0, color='firebrick',  lw=2, label=r'$E_\mathrm{therm}$')
    if RUNS[label]['has_cr']:
        ax.plot(t, e['ecr'] / E0, color='forestgreen', lw=2,
                ls='--', label=r'$E_\mathrm{CR}$')
    ax.plot(t, e['etotal'] / E0, color='k', lw=1.5,
            ls=':', alpha=0.7, label=r'$E_\mathrm{total}$')

    ax.set_title(label.replace('\n', '\n'), fontsize=10)
    ax.set_xlabel('Time [Myr]', fontsize=10)
    ax.set_ylabel(r'Energy $/ E_0$', fontsize=10)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.25, ls='--')
    ax.set_xlim(left=0)

fig.suptitle('Energy components (normalised to initial total energy)', fontsize=13)
plt.tight_layout()
fname = os.path.join(OUTDIR, 'energy_components.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 – CR energy fraction f_CR = E_CR / E_total  (CR runs only)
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 4: CR energy fraction …')

cr_labels = [lb for lb in RUNS if RUNS[lb]['has_cr']]

fig, ax = plt.subplots(figsize=(8, 5))

for label in cr_labels:
    e   = energy[label]
    t   = e['time_myr']
    with np.errstate(divide='ignore', invalid='ignore'):
        f_cr = np.where(e['etotal'] > 0, e['ecr'] / e['etotal'], np.nan)
    ax.plot(t, f_cr * 100., color=COLORS[label], lw=2,
            label=label.replace('\n', ' '))

ax.set_xlabel('Time [Myr]', fontsize=12)
ax.set_ylabel(r'$f_\mathrm{CR} = E_\mathrm{CR}/E_\mathrm{total}\;\;[\%]$', fontsize=12)
ax.set_title('Fraction of total energy in cosmic rays', fontsize=13)
ax.legend(fontsize=9, framealpha=0.7)
ax.grid(True, alpha=0.3, ls='--')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'cr_fraction.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 – X_cr = P_cr / P_th  radial profile at each snapshot (CR runs)
# Computes from snapshot data:
#   P_th = (γ  − 1) · ρ · u
#   P_cr = (γ_cr−1) · ρ · cren
#   X_cr = P_cr / P_th
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 5: X_cr = P_cr/P_th radial profiles …')


def _log_radial_mean(pos, vals, ctr, r_range=(5., 200.), nbins=200):
    """Logarithmically binned radial mean. Returns (r_ctrs, profile)."""
    r = np.linalg.norm(pos - ctr, axis=1)
    r_lo, r_hi = r_range
    mask = (r >= r_lo) & (r <= r_hi) & np.isfinite(vals)
    r_edges = np.logspace(np.log10(r_lo), np.log10(r_hi), nbins + 1)
    r_ctrs  = 0.5 * (r_edges[:-1] + r_edges[1:])
    idx     = np.digitize(r[mask], r_edges)
    profile = np.array([
        vals[mask][idx == i].mean() if np.any(idx == i) else np.nan
        for i in range(1, nbins + 1)
    ])
    return r_ctrs, profile


import matplotlib.cm as cm

# determine number of snapshots for each CR run
def _count_snaps(path):
    n = 0
    while os.path.exists(os.path.join(path, f'snap_{n:03d}.hdf5')):
        n += 1
    return n

N_CR    = _count_snaps(BASE + '/output_cr/')
N_CREXP = _count_snaps(BASE + '/output_crexps/')
print(f'  output_cr: {N_CR} snaps, output_crexps: {N_CREXP} snaps')

R_RANGE = (5., 200.)   # kpc
NBINS   = 300

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

run_info = [
    ('CR (no diff)\n(output_cr)',  BASE+'/output_cr/',    N_CR,    'Blues',   axes[0]),
    ('CR (diff)\n(output_crexps)', BASE+'/output_crexps/',N_CREXP, 'Oranges', axes[1]),
]

for label, path, n_snaps, cmap_name, ax in run_info:
    if n_snaps == 0:
        ax.text(0.5, 0.5, 'No snapshots', ha='center', va='center',
                transform=ax.transAxes)
        continue

    cmap = cm.get_cmap(cmap_name, max(n_snaps, 2))
    run  = arun.Run(snappath=path, snapbase=SNAPBASE)

    for i in range(n_snaps):
        try:
            s = run.loadSnap(snapnum=i)
        except Exception as exc:
            print(f'    Could not load {label} snap {i}: {exc}')
            continue

        if 'cren' not in s.data:
            print(f'    Snap {i} has no cren field – skipping X_cr')
            continue

        rho  = s.data['rho']    # code density
        u_   = s.data['u']      # code specific internal energy
        cren = s.data['cren']   # code specific CR energy

        p_th = (GAMMA    - 1.) * rho * u_
        p_cr = (GAMMA_CR - 1.) * rho * cren

        with np.errstate(divide='ignore', invalid='ignore'):
            x_cr = np.where(p_th > 0, p_cr / p_th, np.nan)

        # get physical time for legend
        t_code = s.header['Time']
        t_myr  = code_time_to_Myr(t_code)

        ctr = np.array([s.boxsize / 2.] * 3)
        r, xcr_prof = _log_radial_mean(s.data['pos'], x_cr, ctr,
                                       r_range=R_RANGE, nbins=NBINS)

        color = cmap(i / max(n_snaps - 1, 1))
        lw    = 1.2 + 0.6 * (i / max(n_snaps - 1, 1))
        ax.plot(r, xcr_prof, color=color, lw=lw, alpha=0.85,
                label=f't = {t_myr:.1f} Myr')

    ax.axhline(1, color='k', lw=0.8, ls='--', alpha=0.4, label=r'$X_{CR}=1$')
    ax.set_title(label.replace('\n', '\n'), fontsize=11)
    ax.set_xlabel('Radius [kpc]', fontsize=11)
    ax.set_ylabel(r'$X_{CR} = P_{CR} / P_\mathrm{th}$', fontsize=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=7, framealpha=0.6, loc='best', ncol=2)
    ax.grid(True, which='both', alpha=0.2, ls='--')

fig.suptitle(r'$X_{CR} = P_{CR}/P_\mathrm{th}$ radial profiles', fontsize=13)
plt.tight_layout()
fname = os.path.join(OUTDIR, 'xcr_radial.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 – CR energy budget: injected vs cooling losses vs stored
# (for CR runs based on crenergy.txt)
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 6: CR energy budget …')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, label in zip(axes, cr_labels):
    if label not in cre or cre[label] is None:
        ax.text(0.5, 0.5, 'crenergy.txt not available',
                ha='center', va='center', transform=ax.transAxes)
        continue
    cr  = cre[label]
    t   = cr['time_myr']
    E0  = cr['ecr'][0] if cr['ecr'][0] != 0 else 1.  # avoid div by 0 at t=0

    ax.plot(t, cr['ecr'],          color='forestgreen', lw=2,  label=r'$E_{CR}$ (stored)')
    ax.plot(t, cr['ecr_inject'],   color='royalblue',   lw=2,  ls='--',
            label=r'$E_{CR,\,\mathrm{injected}}$')
    ax.plot(t, np.abs(cr['ecr_hadronic']), color='firebrick', lw=1.5, ls=':',
            label=r'$|E_{CR,\,\mathrm{hadronic}}|$')
    ax.plot(t, np.abs(cr['ecr_coulomb']),  color='darkorange', lw=1.5, ls='-.',
            label=r'$|E_{CR,\,\mathrm{Coulomb}}|$')

    ax.set_title(label.replace('\n', ' '), fontsize=11)
    ax.set_xlabel('Time [Myr]', fontsize=11)
    ax.set_ylabel('CR energy [code units]', fontsize=11)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.25, ls='--')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

fig.suptitle('CR energy budget over time', fontsize=13)
plt.tight_layout()
fname = os.path.join(OUTDIR, 'cr_energy_budget.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 – Combined summary: X_cr and f_CR on one figure
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 7: combined X_cr + f_CR summary …')

fig, (ax_xcr, ax_fcr) = plt.subplots(1, 2, figsize=(14, 5))

# Left: f_CR vs time for both CR runs
for label in cr_labels:
    e   = energy[label]
    t   = e['time_myr']
    with np.errstate(divide='ignore', invalid='ignore'):
        f_cr = np.where(e['etotal'] > 0, e['ecr'] / e['etotal'], np.nan)
    ax_fcr.plot(t, f_cr * 100., color=COLORS[label], lw=2,
                label=label.replace('\n', ' '))

ax_fcr.set_xlabel('Time [Myr]', fontsize=12)
ax_fcr.set_ylabel(r'$f_{CR} = E_{CR}/E_\mathrm{total}\;\;[\%]$', fontsize=12)
ax_fcr.set_title('CR energy fraction vs. time', fontsize=12)
ax_fcr.legend(fontsize=9, framealpha=0.7)
ax_fcr.grid(True, alpha=0.3, ls='--')
ax_fcr.set_xlim(left=0)
ax_fcr.set_ylim(bottom=0)

# Right: X_cr vs radius at last snapshot for both CR runs
for label, path, n_snaps, cmap_name, ls_style in [
        ('CR (no diff)\n(output_cr)',  BASE+'/output_cr/',    N_CR,    'Blues',   '-'),
        ('CR (diff)\n(output_crexps)', BASE+'/output_crexps/',N_CREXP, 'Oranges', '--'),
]:
    if n_snaps == 0:
        continue
    run = arun.Run(snappath=path, snapbase=SNAPBASE)
    try:
        s = run.loadSnap(snapnum=n_snaps - 1)
    except Exception as exc:
        print(f'    Could not load {label} last snap: {exc}')
        continue
    if 'cren' not in s.data:
        continue

    rho  = s.data['rho']
    u_   = s.data['u']
    cren = s.data['cren']
    p_th = (GAMMA    - 1.) * rho * u_
    p_cr = (GAMMA_CR - 1.) * rho * cren
    with np.errstate(divide='ignore', invalid='ignore'):
        x_cr = np.where(p_th > 0, p_cr / p_th, np.nan)

    t_myr = code_time_to_Myr(s.header['Time'])
    ctr   = np.array([s.boxsize / 2.] * 3)
    r, xcr_prof = _log_radial_mean(s.data['pos'], x_cr, ctr,
                                   r_range=R_RANGE, nbins=NBINS)

    ax_xcr.plot(r, xcr_prof, color=COLORS[label], lw=2, ls=ls_style,
                label=f'{label.replace(chr(10), " ")}  (t={t_myr:.1f} Myr)')

ax_xcr.axhline(1, color='k', lw=0.8, ls=':', alpha=0.5, label=r'$X_{CR}=1$')
ax_xcr.set_xscale('log')
ax_xcr.set_yscale('log')
ax_xcr.set_xlabel('Radius [kpc]', fontsize=12)
ax_xcr.set_ylabel(r'$X_{CR} = P_{CR}/P_\mathrm{th}$', fontsize=12)
ax_xcr.set_title(r'$X_{CR}$ at final snapshot', fontsize=12)
ax_xcr.legend(fontsize=9, framealpha=0.7)
ax_xcr.grid(True, which='both', alpha=0.25, ls='--')

fig.suptitle(r'CR energy summary', fontsize=13)
plt.tight_layout()
fname = os.path.join(OUTDIR, 'cr_summary.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8 – Cross-simulation check: E_mech (no CR) vs E_mech+CR
# Overlays emech for all three runs AND etotal (incl. CR) for CR runs,
# all on the same axes so deviations between runs are immediately visible.
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 8: cross-simulation energy comparison …')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax_abs, ax_rel = axes

# Reference: no-CR emech at t=0
E0_ref = energy['no-CR\n(output2)']['emech'][0]

for label, e in energy.items():
    t    = e['time_myr']
    col  = COLORS[label]
    name = label.replace('\n', ' ')

    # ── E_mech (thermal + kinetic + potential) ──
    ax_abs.plot(t, e['emech'] / E0_ref, color=col, lw=2, ls='-',
                label=f'{name}  $E_\\mathrm{{mech}}$')

    # ── E_total (= E_mech + E_CR where applicable) ──
    if RUNS[label]['has_cr']:
        ax_abs.plot(t, e['etotal'] / E0_ref, color=col, lw=1.5, ls='--',
                    label=f'{name}  $E_\\mathrm{{total}}$')

# relative difference of emech w.r.t. no-CR run
e_nocr = energy['no-CR\n(output2)']
for label, e in energy.items():
    if label == 'no-CR\n(output2)':
        continue
    col  = COLORS[label]
    name = label.replace('\n', ' ')
    # interpolate no-CR emech onto this run's time grid for a clean diff
    emech_nocr_interp = np.interp(e['time_myr'], e_nocr['time_myr'], e_nocr['emech'])
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_mech  = (e['emech']  - emech_nocr_interp) / np.abs(emech_nocr_interp) * 100.
        rel_total = (e['etotal'] - emech_nocr_interp) / np.abs(emech_nocr_interp) * 100.

    ax_rel.plot(e['time_myr'], rel_mech,  color=col, lw=2, ls='-',
                label=f'{name}  $\\Delta E_\\mathrm{{mech}}/E_0$')
    ax_rel.plot(e['time_myr'], rel_total, color=col, lw=1.5, ls='--',
                label=f'{name}  $\\Delta E_\\mathrm{{total}}/E_0$')

ax_abs.axhline(1, color='k', lw=0.8, ls=':', alpha=0.5)
ax_abs.set_xlabel('Time [Myr]', fontsize=12)
ax_abs.set_ylabel(r'Energy $/ E_\mathrm{mech,0}^{\,\mathrm{no-CR}}$', fontsize=12)
ax_abs.set_title('Absolute energies across all runs\n'
                 r'solid: $E_\mathrm{mech}$  /  dashed: $E_\mathrm{total}$', fontsize=11)
ax_abs.legend(fontsize=8, framealpha=0.7)
ax_abs.grid(True, alpha=0.3, ls='--')
ax_abs.set_xlim(left=0)

ax_rel.axhline(0, color='k', lw=0.8, ls=':', alpha=0.5)
ax_rel.set_xlabel('Time [Myr]', fontsize=12)
ax_rel.set_ylabel(r'$(E - E_\mathrm{mech}^\mathrm{no-CR}) / E_0\;\;[\%]$', fontsize=12)
ax_rel.set_title('Relative difference w.r.t. no-CR $E_\\mathrm{mech}$\n'
                 r'solid: $\Delta E_\mathrm{mech}$  /  dashed: $\Delta E_\mathrm{total}$',
                 fontsize=11)
ax_rel.legend(fontsize=8, framealpha=0.7)
ax_rel.grid(True, alpha=0.3, ls='--')
ax_rel.set_xlim(left=0)

fig.suptitle('Cross-simulation energy comparison: with vs. without CRs', fontsize=13)
plt.tight_layout()
fname = os.path.join(OUTDIR, 'energy_crosssim.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')


# ─────────────────────────────────────────────────────────────────────────────
# NFW potential (matching Arepo's grav_external.c exactly)
# Config: STATICNFW  NFW_C=7  NFW_M200=100.0  NFW_Eps=0.01
#         NFW_DARKFRACTION=0.844  NFW_h=0.7
# Units:  UnitLength = 3.08568e21 cm (kpc)
#         UnitMass   = 1.989e43 g   (1e10 Msun)
#         UnitVel    = 1e5 cm/s     (km/s)
# ─────────────────────────────────────────────────────────────────────────────
_GRAVITY_CGS = 6.6738e-8        # cm³ g⁻¹ s⁻²
_HUBBLE_CGS  = 3.2407789e-18    # h/s
_UNIT_M      = 1.989e43         # g
_UNIT_T_s    = _UNIT_L / _UNIT_V  # s

# G in code units
G_CODE = _GRAVITY_CGS / (_UNIT_L**3) * _UNIT_M * (_UNIT_T_s**2)

# Hubble in code units (* h)
H_CODE = _HUBBLE_CGS * _UNIT_T_s          # Hubble / h in code
H_CODE *= 0.7                             # NFW_h = 0.7

# NFW parameters
_NFW_C   = 7.0
_NFW_M200 = 100.0      # code mass units (= 1e12 Msun)
_NFW_Eps  = 0.01
_NFW_DARKFRACTION = 0.844

# Derived quantities (mirroring init_static_nfw)
_R200 = (_NFW_M200 * G_CODE / (100.0 * H_CODE**2))**(1.0/3.0)
_Rs   = _R200 / _NFW_C
_Dc   = 200.0/3.0 * _NFW_C**3 / (np.log(1 + _NFW_C) - _NFW_C / (1.0 + _NFW_C))
_RhoCrit = 3.0 * H_CODE**2 / (8.0 * np.pi * G_CODE)

# Normalisation factor (Fac)
def _enclosed_mass_raw(R, Fac=1.0):
    """Arepo's enclosed_mass_nfw with Fac=1 (for calibration)."""
    Rc = min(R, _Rs * _NFW_C)   # truncated at R200
    eps = _NFW_Eps
    rs = _Rs
    term0 = -(rs**3 * (1 - eps + np.log(rs) - 2*eps*np.log(rs) +
              eps**2 * np.log(eps * rs))) / ((eps - 1)**2)
    term1 = (rs**3 * (rs - eps*rs - (2*eps - 1)*(Rc + rs)*np.log(Rc + rs) +
              eps**2 * (Rc + rs) * np.log(Rc + eps*rs))) / \
             ((eps - 1)**2 * (Rc + rs))
    return Fac * 4 * np.pi * _RhoCrit * _Dc * (term0 + term1)

# Calibrate Fac so that M_enclosed(R200) = M200
_V200 = 10.0 * H_CODE * _R200
_Fac_NFW = _V200**3 / (10.0 * G_CODE * H_CODE) / _enclosed_mass_raw(_R200, Fac=1.0)


def nfw_enclosed_mass(R):
    """Enclosed NFW mass at radius R (code units), truncated at R200."""
    Rc = np.minimum(R, _Rs * _NFW_C)
    eps = _NFW_Eps
    rs = _Rs
    term0 = -(rs**3 * (1 - eps + np.log(rs) - 2*eps*np.log(rs) +
              eps**2 * np.log(eps * rs))) / ((eps - 1)**2)
    term1 = (rs**3 * (rs - eps*rs - (2*eps - 1)*(Rc + rs)*np.log(Rc + rs) +
              eps**2 * (Rc + rs) * np.log(Rc + eps*rs))) / \
             ((eps - 1)**2 * (Rc + rs))
    return _Fac_NFW * 4 * np.pi * _RhoCrit * _Dc * (term0 + term1) * _NFW_DARKFRACTION


def nfw_potential_at_r(r):
    """
    NFW gravitational potential at radius r (scalar or array, code units).
    Φ(r) = -G ∫_r^∞ M(<r')/r'² dr'
    For truncated NFW (at R200), Φ(r) = -G M(<r)/r  for r >= R200
    and for r < R200 we integrate numerically for accuracy.

    Simpler and standard: Φ(r) = -G M200 / [r · g(c)] · ln(1 + r/Rs)
    But Arepo uses the softened formula, so we do numerical integration
    to match exactly.
    """
    from scipy.integrate import quad
    r = np.atleast_1d(r).astype(float)
    phi = np.zeros_like(r)
    for i, ri in enumerate(r):
        if ri <= 0:
            phi[i] = np.nan
            continue
        # Φ(r) = -G ∫_r^∞ M(<r')/r'² dr'
        # Split: [r, R200] uses enclosed_mass, beyond R200 M is constant = M(R200)
        r200 = _Rs * _NFW_C
        m200 = nfw_enclosed_mass(r200)
        if ri >= r200:
            phi[i] = -G_CODE * m200 / ri
        else:
            def integrand(rp):
                return nfw_enclosed_mass(rp) / rp**2
            val, _ = quad(integrand, ri, r200)
            phi[i] = -G_CODE * val - G_CODE * m200 / r200
    return phi


def nfw_epot_snapshot(s):
    """Compute total NFW gravitational potential energy for a snapshot.
    E_pot = Σ m_i · Φ_NFW(r_i)
    """
    pos  = s.data['pos']
    mass = s.data['mass']
    ctr  = np.array([s.boxsize / 2.] * 3)
    r    = np.linalg.norm(pos - ctr, axis=1)

    # For many particles, vectorised integration is slow.
    # Use the analytic NFW potential:
    #   Φ(r) = -G · M200 / [ln(1+c) - c/(1+c)] · ln(1 + r/Rs) / r
    # This matches the standard NFW (softening eps→0 limit).
    # Since NFW_Eps=0.01 ≈ 0, this is an excellent approximation.
    gc = np.log(1 + _NFW_C) - _NFW_C / (1.0 + _NFW_C)
    M200_dark = _NFW_M200 * _NFW_DARKFRACTION
    # For r > R200, truncate: Φ(r) = -G M200_dark / r
    r200 = _Rs * _NFW_C
    phi = np.where(
        r > 0,
        np.where(
            r <= r200,
            -G_CODE * M200_dark / gc * np.log(1.0 + r / _Rs) / np.maximum(r, 1e-30),
            -G_CODE * M200_dark / r
        ),
        0.0
    )
    return np.sum(mass * phi)


print(f'  NFW potential: R200 = {_R200:.2f} kpc,  Rs = {_Rs:.2f} kpc,  '
      f'G_code = {G_CODE:.4e},  M200*f_dark = {_NFW_M200*_NFW_DARKFRACTION:.2f} [1e10 Msun]')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 9 – Total energy computed directly from snapshots
# Sums  E_kin  = Σ ½ m v²
#       E_th   = Σ m u
#       E_CR   = Σ m cren        (if present)
#       E_pot  = Σ m Φ_NFW(r)    (external NFW potential)
# and compares across simulations + against energy.txt
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 9: total energy from snapshots (including NFW potential) …')

snap_energy = {}

for label, cfg in RUNS.items():
    path    = cfg['path']
    has_cr  = cfg['has_cr']
    n_snaps = _count_snaps(path)
    if n_snaps == 0:
        print(f'  {label.replace(chr(10)," ")}: no snapshots found – skipping')
        continue

    run = arun.Run(snappath=path, snapbase=SNAPBASE)

    times_code = []
    ekin_arr   = []
    etherm_arr = []
    ecr_arr    = []
    epot_arr   = []
    etot_arr   = []

    for i in range(n_snaps):
        try:
            s = run.loadSnap(snapnum=i)
        except Exception as exc:
            print(f'    Could not load {label.replace(chr(10)," ")} snap {i}: {exc}')
            continue

        mass = s.data['mass']
        vel  = s.data['vel']     # (N, 3)
        u_   = s.data['u']      # specific internal energy

        ek = 0.5 * np.sum(mass * np.sum(vel**2, axis=1))
        et = np.sum(mass * u_)

        ec = 0.0
        if has_cr and 'cren' in s.data:
            ec = np.sum(mass * s.data['cren'])

        ep = nfw_epot_snapshot(s)

        times_code.append(s.header['Time'])
        ekin_arr.append(ek)
        etherm_arr.append(et)
        ecr_arr.append(ec)
        epot_arr.append(ep)
        etot_arr.append(ek + et + ec - ep)

    snap_energy[label] = {
        'time':     np.array(times_code),
        'time_myr': code_time_to_Myr(np.array(times_code)),
        'ekin':     np.array(ekin_arr),
        'etherm':   np.array(etherm_arr),
        'ecr':      np.array(ecr_arr),
        'epot':     np.array(epot_arr),
        'etotal':   np.array(etot_arr),
        'emech':    np.array(ekin_arr) + np.array(etherm_arr) + np.array(epot_arr),
    }
    print(f'  {label.replace(chr(10)," "):<35s}: {n_snaps} snaps, '
          f'E_tot(snap0) = {etot_arr[0]:.4g},  E_pot(snap0) = {epot_arr[0]:.4g} [code]')


# ── 9a: Normalised total energy from snapshots across all runs ──
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

ax_norm  = axes[0, 0]
ax_dev   = axes[0, 1]
ax_comp  = axes[0, 2]
ax_cross = axes[1, 0]
ax_parts = axes[1, 1]
ax_epot  = axes[1, 2]

# Panel (a): E_total(t) / E_total(0) from snapshots (WITH E_pot)
for label in snap_energy:
    se = snap_energy[label]
    E0 = se['etotal'][0]
    ax_norm.plot(se['time_myr'], se['etotal']/E0, 'o-', color=COLORS[label],
                 ms=4, lw=1.5, label=label.replace('\n', ' '))

ax_norm.axhline(1, color='k', lw=0.8, ls='--', alpha=0.5)
ax_norm.set_xlabel('Time [Myr]', fontsize=11)
ax_norm.set_ylabel(r'$E_\mathrm{total}(t)/E_\mathrm{total}(0)$', fontsize=11)
ax_norm.set_title(r'(a) Total energy (incl. $\Phi_\mathrm{NFW}$) normalised', fontsize=11)
ax_norm.legend(fontsize=8, framealpha=0.7)
ax_norm.grid(True, alpha=0.3, ls='--')
ax_norm.set_xlim(left=0)

# Panel (b): Fractional deviation from snapshots
for label in snap_energy:
    se = snap_energy[label]
    E0 = se['etotal'][0]
    dE = (se['etotal'] - E0) / np.abs(E0) * 100.
    ax_dev.plot(se['time_myr'], dE, 'o-', color=COLORS[label],
                ms=4, lw=1.5, label=label.replace('\n', ' '))

ax_dev.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
ax_dev.set_xlabel('Time [Myr]', fontsize=11)
ax_dev.set_ylabel(r'$\Delta E/E_0\;\;[\%]$', fontsize=11)
ax_dev.set_title(r'(b) Fractional deviation $\Delta E/E_0$ (incl. $\Phi_\mathrm{NFW}$)', fontsize=11)
ax_dev.legend(fontsize=8, framealpha=0.7)
ax_dev.grid(True, alpha=0.3, ls='--')
ax_dev.set_xlim(left=0)

# Panel (c): Snapshot vs energy.txt comparison for each run
for label in snap_energy:
    se = snap_energy[label]
    e  = energy[label]
    col = COLORS[label]
    name = label.replace('\n', ' ')

    # energy.txt total (does NOT include E_pot since COMPUTE_POTENTIAL_ENERGY is off)
    ax_comp.plot(e['time_myr'], e['etotal'], '-', color=col, lw=2,
                 alpha=0.6, label=f'{name} energy.txt (no $\\Phi$)')
    # snapshot total (includes E_pot)
    ax_comp.plot(se['time_myr'], se['etotal'], 'o', color=col, ms=5,
                 mfc='none', mew=1.5, label=f'{name} snap (with $\\Phi$)')

ax_comp.set_xlabel('Time [Myr]', fontsize=11)
ax_comp.set_ylabel('Total energy [code units]', fontsize=11)
ax_comp.set_title('(c) Snapshot energy vs energy.txt', fontsize=11)
ax_comp.legend(fontsize=7, framealpha=0.7, ncol=2)
ax_comp.grid(True, alpha=0.3, ls='--')
ax_comp.set_xlim(left=0)

# Panel (d): Cross-simulation comparison – all on same axes
if 'no-CR\n(output2)' in snap_energy:
    E0_ref_snap = snap_energy['no-CR\n(output2)']['etotal'][0]
else:
    E0_ref_snap = 1.0

for label in snap_energy:
    se  = snap_energy[label]
    col = COLORS[label]
    name = label.replace('\n', ' ')

    ax_cross.plot(se['time_myr'], se['etotal'] / E0_ref_snap, 'o-',
                  color=col, ms=4, lw=1.5,
                  label=f'{name}  $E_\\mathrm{{total}}$')

ax_cross.axhline(1, color='k', lw=0.8, ls=':', alpha=0.5)
ax_cross.set_xlabel('Time [Myr]', fontsize=11)
ax_cross.set_ylabel(r'$E_\mathrm{total} / E_\mathrm{total,0}^{\,\mathrm{no-CR}}$', fontsize=11)
ax_cross.set_title(r'(d) Cross-simulation $E_\mathrm{total}$ (incl. $\Phi_\mathrm{NFW}$)', fontsize=10)
ax_cross.legend(fontsize=7, framealpha=0.7, ncol=1)
ax_cross.grid(True, alpha=0.3, ls='--')
ax_cross.set_xlim(left=0)

# Panel (e): Energy components breakdown (one example: first CR run)
for label in snap_energy:
    se  = snap_energy[label]
    E0  = np.abs(se['etotal'][0])
    col = COLORS[label]
    name = label.replace('\n', ' ')
    ax_parts.plot(se['time_myr'], se['ekin']   / E0, 's-',  color=col, ms=3, lw=1,
                  alpha=0.6, label=f'{name} $E_\\mathrm{{kin}}$')
    ax_parts.plot(se['time_myr'], se['etherm'] / E0, '^-',  color=col, ms=3, lw=1,
                  alpha=0.6, label=f'{name} $E_\\mathrm{{th}}$')
    ax_parts.plot(se['time_myr'], se['epot']   / E0, 'v--', color=col, ms=3, lw=1,
                  alpha=0.6, label=f'{name} $E_\\mathrm{{pot}}$')

ax_parts.set_xlabel('Time [Myr]', fontsize=11)
ax_parts.set_ylabel(r'Energy $/ |E_\mathrm{total,0}|$', fontsize=11)
ax_parts.set_title('(e) Energy components from snapshots', fontsize=10)
ax_parts.legend(fontsize=6, framealpha=0.6, ncol=2)
ax_parts.grid(True, alpha=0.3, ls='--')
ax_parts.set_xlim(left=0)

# Panel (f): E_pot evolution for all runs
for label in snap_energy:
    se  = snap_energy[label]
    col = COLORS[label]
    name = label.replace('\n', ' ')
    ax_epot.plot(se['time_myr'], se['epot'], 'o-', color=col, ms=4, lw=1.5,
                 label=name)

ax_epot.set_xlabel('Time [Myr]', fontsize=11)
ax_epot.set_ylabel(r'$E_\mathrm{pot,NFW}$ [code units]', fontsize=11)
ax_epot.set_title(r'(f) NFW potential energy $\sum m_i \Phi_\mathrm{NFW}(r_i)$', fontsize=10)
ax_epot.legend(fontsize=8, framealpha=0.7)
ax_epot.grid(True, alpha=0.3, ls='--')
ax_epot.set_xlim(left=0)

fig.suptitle(r'Energy conservation from snapshot data (including NFW $\Phi$)', fontsize=14, y=1.01)
plt.tight_layout()
fname = os.path.join(OUTDIR, 'energy_from_snapshots.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')


print('\n✓  All plots saved to:', OUTDIR)
