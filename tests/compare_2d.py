"""
compare_2d.py
=============
Side-by-side 2D slice comparisons of:
  - output_cr     (no CR diffusion)
  - output_crexps (with CR diffusion)

For each snapshot shared between the two runs, this script creates one PNG
with rows = physical quantities and columns = simulations.  The shared
colour scale per row lets you immediately see structural differences.

Quantities shown
----------------
  rho       : gas density
  pres      : thermal pressure
  crpres    : cosmic-ray pressure  (derived from cren)
  temp      : gas temperature      (derived from u)
  speed     : total speed          (|vel|)
  bfldenerg : magnetic energy density (|B|^2 / 2rho)

Saved to: test_ai/plots/
"""

import sys, os

# ── paths ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/arepo-snap-util')
sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/gasCloudNfw/snap-plotting')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as u
import arepo_run as arun

# ── physical constants ────────────────────────────────────────────────────────
BASE_PATH  = '/cosma8/data/dp317/dc-naza3/gasCloudNfw'
SNAPBASE   = 'snap_'
GAMMA      = 5./3
GAMMA_CR   = 4./3
XH         = 0.76          # hydrogen mass fraction
k_B        = 1.381e-16     # erg/K
m_p        = 1.66e-24      # g
unit_v     = 1.e5          # cm/s

# output directory (next to this script)
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(OUTDIR, exist_ok=True)

# ── derived-field calculator ──────────────────────────────────────────────────
def enrich_snap(s):
    """Attach commonly needed derived fields to s.data in-place."""
    # mean molecular weight (assume fully ionised, no ne field needed)
    mu = 0.6 * m_p
    s.data['temp']  = (GAMMA - 1) * mu / k_B * s.data['u'] * unit_v**2
    s.data['speed'] = np.linalg.norm(s.data['vel'], axis=1)

    if 'cren' in s.data:
        # CR pressure:  P_cr = (gamma_cr - 1) * rho * e_cr
        s.data['crpres'] = (GAMMA_CR - 1) * s.data['rho'] * s.data['cren']

    if 'bfld' in s.data:
        B2 = np.sum(s.data['bfld']**2, axis=1)
        s.data['bfldenerg'] = B2 / (2.0 * s.data['rho'])

    return s


def load_snap(snappath, num):
    o = arun.Run(snappath=snappath, snapbase=SNAPBASE)
    s = o.loadSnap(snapnum=num)
    return enrich_snap(s)


def snap_time_myr(s):
    Lu = s.header['UnitLength_in_cm'] * u.cm
    Vu = s.header['UnitVelocity_in_cm_per_s'] * u.cm / u.s
    return (s.header['Time'] * Lu / Vu).to(u.Myr).value


# ── quantity catalogue ────────────────────────────────────────────────────────
# (field_key,  row_label,            cmap,       log?,  vrange_or_None)
QUANTITIES = [
    ('rho',       r'Density [$\rho$]',           'inferno',  True,  [1e-7, 1e-2]),
    ('pres',      r'Thermal Pressure',            'gnuplot',  True,  [1e-4, 1e1 ]),
    ('crpres',    r'CR Pressure',                 'viridis',  True,  [1e-4, 1e1 ]),
    ('temp',      r'Temperature [K]',             'plasma',   True,  [1e3,  1e7 ]),
    ('speed',     r'Speed [code units]',          'cividis',  True,  [1e-3, 1e3 ]),
    ('bfldenerg', r'Mag. Energy Density',         'Blues_r',  True,  None        ),
]

# ── plotting settings ─────────────────────────────────────────────────────────
PLOTSIZE = 50       # kpc per panel half-width
RES      = 512      # pixels per panel
AXES_XZ  = [0, 2]  # project onto x-z plane (side view)

# Snapshots present in both runs
PATH_CR    = BASE_PATH + '/output_cr/'
PATH_CREXP = BASE_PATH + '/output_crexps/'
COMMON_SNAPS = range(6)   # snaps 000–005 exist in both

# ── main loop ─────────────────────────────────────────────────────────────────
# Panel geometry
# Each simulation panel is square (PLOTSIZE × PLOTSIZE kpc).
# Layout per row: [panel_cr | panel_crexp | colorbar]
# width_ratios [1, 1, 0.05] keeps both data panels equal and colourbar narrow.
PANEL_IN = 5.0   # inches per square panel

for snap_num in COMMON_SNAPS:
    print(f'Processing snapshot {snap_num:03d} ...')

    s_cr   = load_snap(PATH_CR,    snap_num)
    s_crex = load_snap(PATH_CREXP, snap_num)
    t_cr   = snap_time_myr(s_cr)
    t_crex = snap_time_myr(s_crex)

    n_qty = len(QUANTITIES)

    # Figure: 2 square panels + thin colorbar column, one row per quantity
    fig_w = PANEL_IN * 2 + 1.4    # ~11.4 inches wide
    fig_h = PANEL_IN * n_qty + 1.2
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = gridspec.GridSpec(
        n_qty, 3,
        figure=fig,
        width_ratios=[1, 1, 0.08],   # wider colorbar column (was 0.05 → invisible)
        hspace=0.10,                  # less gap between rows → taller panels → less vertical compression
        wspace=0.06,
        left=0.07, right=0.96,
        top=0.96, bottom=0.03,
    )

    fig.suptitle(
        f'Snap {snap_num:03d}   |   No diffusion: {t_cr:.1f} Myr'
        f'   |   With diffusion: {t_crex:.1f} Myr',
        fontsize=13,
    )

    plot_centre = [s_cr.boxsize / 2] * 3

    for row, (field, label, cmap, logplot, vrange) in enumerate(QUANTITIES):
        ax_cr   = fig.add_subplot(gs[row, 0])
        ax_crex = fig.add_subplot(gs[row, 1])
        ax_cb   = fig.add_subplot(gs[row, 2])

        # Column headers on first row only
        if row == 0:
            ax_cr.set_title('output_cr  (no CR diffusion)',       fontsize=10)
            ax_crex.set_title('output_crexps  (with CR diffusion)', fontsize=10)

        last_mappable = None

        for ax, s in [(ax_cr, s_cr), (ax_crex, s_crex)]:
            if field not in s.data:
                ax.text(0.5, 0.5, f'{field}\nnot available',
                        ha='center', va='center', transform=ax.transAxes, fontsize=9)
                ax.set_axis_off()
                continue

            try:
                # colorbar=False — we draw it ourselves in ax_cb
                s.axplot_Aslice(
                    ax,
                    value=field,
                    cmap=cmap,
                    colorbar=False,
                    vrange=vrange,
                    axes=AXES_XZ,
                    logplot=logplot,
                    box=[PLOTSIZE, PLOTSIZE],
                    center=plot_centre,
                    proj=False,
                    proj_fact=0.1,
                    res=RES,
                )

                # Force square aspect so the slice is not distorted
                ax.set_aspect('equal', adjustable='box')

                # Shift tick labels to be relative to box centre
                bh = s.boxsize / 2
                xtks = ax.get_xticks()
                ytks = ax.get_yticks()
                ax.set_xticklabels([f'{float(t) - bh:.0f}' for t in xtks], fontsize=7)
                ax.set_yticklabels([f'{float(t) - bh:.0f}' for t in ytks], fontsize=7)
                ax.set_xlabel('x [kpc]', fontsize=8)

                # Grab the last drawn collection (pcolormesh) as the mappable
                if ax.collections:
                    last_mappable = ax.collections[-1]
                elif ax.images:
                    last_mappable = ax.images[-1]

            except Exception as exc:
                ax.text(0.5, 0.5, f'Error:\n{exc}',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=7, color='red')

        # Row label on left panel only
        ax_cr.set_ylabel(label, fontsize=9)

        # Shared colorbar for this row
        if last_mappable is not None:
            cb = fig.colorbar(last_mappable, cax=ax_cb)
            cb.ax.tick_params(labelsize=7)
            cb.set_label(label, fontsize=8)
        else:
            ax_cb.set_visible(False)

    fname = os.path.join(OUTDIR, f'compare_2d_snap{snap_num:03d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {fname}')

print('Done.')
