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
BASE_PATH  = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion'
SNAPBASE   = 'snap_'
GAMMA      = 5./3
GAMMA_CR   = 4./3
k_B        = 1.381e-16
m_p        = 1.66e-24
HYDROGENMASS_FRAC = 0.76

PATH_NOCR = BASE_PATH + '/old/output_homo/'
PATH_CR   = BASE_PATH + '/old/output_cr600/'
N_SNAPS   = 3

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clean')
os.makedirs(OUTDIR, exist_ok=True)

from lib import *

# ── quantity catalogue ────────────────────────────────────────────────────────
# (field_key,   label,                   log_y?)
QUANTITIES = [
    ('rho',       r'Density $\rho$ [code]',      True),
    ('pres',      r'Thermal pressure [code]',     True),
    ('crpres',    r'CR pressure [code]',          True),
    ('cren',      r'CR energy density [code]',    True),
    ('xcr',       r'CR pressure fraction',        True),
    ('u',         r'Internal energy [code]',      True),
    ('speed',     r'Speed |v| [code km/s]',       True),
    ('mach',      r'Mach number',                 True),
    ('vrad',      r'Radial velocity [code km/s]', False),
]

R_RANGE = (1e-3, 1)   # kpc
NBINS   = 500

# ── load ALL snapshots for both runs ─────────────────────────────────────────
print('Loading all snapshots...')

snaps_nocr, snaps_cr = [], []
times_nocr, times_cr = [], []

for n in range(N_SNAPS):
    try:
        s = load_snap_data(n, snappath=PATH_NOCR, snapbase=SNAPBASE)
        snaps_nocr.append(s)
        times_nocr.append(calc_snap_time(s))
        print(f'  output_nocr snap {n:03d}  {times_nocr[-1]:.1f} Myr')
    except Exception as e:
        print(f'  output_nocr snap {n:03d} missing.')

for n in range(len(snaps_nocr)):
    try:
        s = load_snap_data(n, snappath=PATH_CR, snapbase=SNAPBASE)
        snaps_cr.append(s)
        times_cr.append(calc_snap_time(s))
        print(f'  output_cr   snap {n:03d}  {times_cr[-1]:.1f} Myr')
    except Exception as e:
        print(f'  output_cr snap {n:03d} missing.')

N_COMMON = min(len(snaps_nocr), len(snaps_cr), N_SNAPS)

# Track position of density peak across snapshots for both runs
print('\nPlotting density peak radius evolution...')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_r, ax_log = axes

def find_density_peak(snaps, times, r_range, nbins):
    rho_peak, r_peak, t_arr = [], [], []
    for n, (s, t) in enumerate(zip(snaps, times)):
        r, rho = radial_profile_lin(s, 'rho', r_range=r_range, nbins=nbins)
        if r is None or not np.any(np.isfinite(rho)):
            continue
        idx = np.nanargmax(rho)
        rho_peak.append(rho[idx])
        r_peak.append(r[idx])
        t_arr.append(t)
    return np.array(t_arr), np.array(rho_peak), np.array(r_peak)

def find_density_shell(snaps, times, r_range, nbins, tol=1e-3):
    rho_shell, t_arr = [], []
    shell_t_arr = []
    for n, (s, t) in enumerate(zip(snaps, times)):
        # s.data['p_dot'] = np.einsum('ij,ij->i',s.data['grar'],s.data['vel'])*s.data['vol']
        # s.data['pr_dot'] = (np.einsum('ij,ij->i',s.data['grar'],s.data['pos'])/s.data['r'])*s.data['vrad']*s.data['vol'] # radial momentum flux grad rho_r * v_rad
        s.data['e_kin'] = 0.5 * s.data['mass'] * s.data['speed']**2
        s.data['e_int'] = s.data['u'] * s.data['mass']
        s.data['e_tot'] = s.data['e_kin'] + s.data['e_int']

        r, rho = radial_profile_lin(s, 'rho', r_range=r_range, nbins=nbins)
        r, vrad = radial_profile_lin(s, 'vrad', r_range=r_range, nbins=nbins)
        r, mass = radial_profile_lin(s, 'mass', r_range=r_range, nbins=nbins)
        r, e_tot = radial_profile_lin(s, 'e_tot', r_range=r_range, nbins=nbins)
        r, e_kin = radial_profile_lin(s, 'e_kin', r_range=r_range, nbins=nbins)
        r, e_int = radial_profile_lin(s, 'e_int', r_range=r_range, nbins=nbins)
        # r, mach = radial_profile_lin(s, 'mach', r_range=r_range, nbins=nbins)
        
        if r is None or not np.any(np.isfinite(rho)):
            continue
            
        idx = np.nanargmax(rho)
        grad = np.abs(np.gradient(rho, r))
        grad_threshold = tol * np.nanmax(grad)
        
        # Expand left from the peak
        left = idx
        while left > 0 and (grad[left] > grad_threshold or left >= idx - 2):
            if left < idx - 2 and grad[left] <= grad_threshold: break
            left -= 1
            
        # Expand right from the peak
        right = idx
        while right < len(rho) - 1 and (grad[right] > grad_threshold or right <= idx + 2):
            if right > idx + 2 and grad[right] <= grad_threshold: break
            right += 1

        # left = 0
        # right = idx
        shell_t = r[right] #- r[left]
        p_r_dot = 4*np.pi*r**2*rho*(vrad)**2
        p_r = mass*vrad
        # if n==2:
        #     plt.plot(r, rho, label='rho')
        #     plt.plot(r[left:right+1], rho[left:right+1], label='rho_limited')
        #     plt.legend()
        avg_rho = np.nanmean(e_int[left:right+1])#np.nanmean(p_r_dot[left:right+1])
        rho_shell.append(avg_rho)
        t_arr.append(t)
        shell_t_arr.append(shell_t)
    return np.array(t_arr), np.array(rho_shell), np.array(shell_t_arr)

t_nocr_arr, rho_nocr_arr, shell_t_nocr_arr = find_density_shell(snaps_nocr, times_nocr, R_RANGE, NBINS)

t_cr_arr,   rho_cr_arr,   shell_t_cr_arr   = find_density_shell(snaps_cr,   times_cr,   R_RANGE, NBINS)

# divide by the initial density to get dimensionless values
norm = False
if len(rho_nocr_arr) > 0 and norm:
    rho0 = rho_nocr_arr[0]
    rho_nocr_arr /= rho0
    rho_cr_arr   /= rho0
t = 0.0
mask = (t_nocr_arr > t) & (t_cr_arr > t)  # exclude early snapshots 
t_nocr_arr = t_nocr_arr[mask]
rho_nocr_arr = rho_nocr_arr[mask]
shell_t_nocr_arr = shell_t_nocr_arr[mask]
t_cr_arr = t_cr_arr[mask]
rho_cr_arr = rho_cr_arr[mask]
shell_t_cr_arr = shell_t_cr_arr[mask]

s = snaps_cr[0]
unit_v = s.header['UnitVelocity_in_cm_per_s']
unit_l = s.header['UnitLength_in_cm'] 
unit_m = s.header['UnitMass_in_g']
unit_t = unit_l / unit_v
unit_rho = unit_m / unit_l**3

# Momentum conversion factor (g cm / s)
unit_p = unit_m * unit_v
unit_e = unit_m * unit_v**2
# Time conversion factor (Myr to seconds)
myr_to_s = 1e6 * 365.25 * 24 * 3600

c=3e10  # cm/s
L_AGN = 1e45  # erg/s
L_AGNc = (L_AGN / c)

# Convert momentum to CGS, time to seconds, then normalize by L_AGNc
p_dot_nocr = (rho_nocr_arr * unit_p) /unit_t #(t_nocr_arr * myr_to_s)
p_dot_cr   = (rho_cr_arr * unit_p) / unit_t#(t_cr_arr * myr_to_s)

e_nocr = (rho_nocr_arr * unit_e) / (t_nocr_arr * myr_to_s)
e_cr   = (rho_cr_arr * unit_e) / (t_cr_arr * myr_to_s)

for ax in axes:
    ax.plot(shell_t_nocr_arr, e_nocr / L_AGN, '^:', color='maroon',
            lw=2, label='no CRs (Hydro only)')
    ax.plot(shell_t_cr_arr,   e_cr / L_AGN,   'o-', color='steelblue',
            lw=2, label='CRs included')
    
    ax.set_xlabel(r'$R_{sh}$ [kpc]', fontsize=11)
    ax.set_ylabel(r'$E_{\rm{int}}/E_{\rm{AGN}}$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, ls='--')

# Add power-law reference lines to the log-log panel
ax_log.set_xscale('log')
ax_log.set_yscale('log')

# axes[0].set_title('Density shell radius vs time (linear)')
# ax_log.set_title('Density shell radius vs time (log–log)')

fig.suptitle('Evolution of density shell position', fontsize=13)
plt.tight_layout()
fname = os.path.join(OUTDIR, 'e_int.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  -> {fname}')

print('\nDone.')

