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

All figures are saved to tests/plots_energy/.
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

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_energy')
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



# determine number of snapshots for each CR run
def _count_snaps(path):
    n = 0
    while os.path.exists(os.path.join(path, f'snap_{n:03d}.hdf5')):
        n += 1
    return n



# # ─────────────────────────────────────────────────────────────────────────────
# # NFW potential (matching Arepo's grav_external.c exactly)
# # Config: STATICNFW  NFW_C=7  NFW_M200=100.0  NFW_Eps=0.01
# #         NFW_DARKFRACTION=0.844  NFW_h=0.7
# # Units:  UnitLength = 3.08568e21 cm (kpc)
# #         UnitMass   = 1.989e43 g   (1e10 Msun)
# #         UnitVel    = 1e5 cm/s     (km/s)
# # ─────────────────────────────────────────────────────────────────────────────
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
            r <= r200,
            -G_CODE * M200_dark / gc * np.log(1.0 + r / _Rs) / np.maximum(r, 1e-30),
            -G_CODE * M200_dark / r
        )
    #phi = nfw_potential_at_r(r)
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
    mass_arr   = []
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

        mtot = np.sum(mass)
        ek = 0.5 * np.sum(mass * np.sum(vel**2, axis=1))
        et = np.sum(mass * u_)

        ec = 0.0
        if has_cr and 'cren' in s.data:
            ec = np.sum(mass * s.data['cren'])

        ep = nfw_epot_snapshot(s)

        times_code.append(s.header['Time'])
        mass_arr.append(mtot)
        ekin_arr.append(ek)
        etherm_arr.append(et)
        ecr_arr.append(ec)
        epot_arr.append(-ep)
        #etot_arr.append(ek + et + ec - ep)

    snap_energy[label] = {
        'time':     np.array(times_code),
        'time_myr': code_time_to_Myr(np.array(times_code)),
        'mass':     np.array(mass_arr),
        'ekin':     np.array(ekin_arr),
        'etherm':   np.array(etherm_arr),
        'ecr':      np.array(ecr_arr),
        'epot':     np.array(epot_arr),
        'etotal':   np.array(ekin_arr) + np.array(etherm_arr) + np.array(epot_arr) + np.array(ecr_arr),
        'emech':    np.array(ekin_arr) + np.array(etherm_arr) + np.array(epot_arr)
    }
    print(f'  {label.replace(chr(10)," "):<35s}: {n_snaps} snaps, '
          f'E_tot(snap0) = {snap_energy[label]["etotal"][0]:.4g},  E_pot(snap0) = {snap_energy[label]["epot"][0]:.4g} [code]')


# ── 9a: Normalised total energy from snapshots across all runs ──
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Panel (a): E_total(t) / E_total(0) from snapshots (WITH E_pot)
for label in snap_energy:
    se = snap_energy[label]
    m0 = se['mass'][0]
    print(f'  {label.replace(chr(10)," ")}: M_total(0) = {m0:.4g} [code]')
    print(se['mass']/m0)
    ax.plot(se['time_myr'][2:], se['mass'][2:]/m0, 'o-', color=COLORS[label],
                 ms=4, lw=1.5, label=label.replace('\n', ' '))

ax.axhline(1, color='k', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Time [Myr]', fontsize=11)
ax.set_ylabel(r'$M_\mathrm{total}(t)/M_\mathrm{total}(0)$', fontsize=11)
ax.set_title(r'Total mass normalised', fontsize=11)
ax.legend(fontsize=8, framealpha=0.7)
ax.grid(True, alpha=0.3, ls='--')
ax.set_xlim(left=0)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'mass_from_snapshots.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')


print('\n✓  All plots saved to:', OUTDIR)