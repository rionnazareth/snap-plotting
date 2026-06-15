import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
import sys
import os
import cmasher

sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/arepo-snap-util')
from tests.lib import *

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
# plt.style.use(['science'])
def column_density_los(s, unit_l):
    sort_idx  = np.argsort(s.data['r'])
    r_sorted  = s.data['r'][sort_idx]
    nH_sorted = s.data['nH_cm'][sort_idx]
    dr_cm     = np.diff(r_sorted, prepend=0) * unit_l   # always ≥ 0 now
    return np.sum(nH_sorted * dr_cm)                    # cm^-2

if __name__ == "__main__":
    print('Running tri-panel density comparison plot...')
    slurm_ntasks = os.getenv('SLURM_NTASKS', '').strip()
    numthreads = int(slurm_ntasks) if slurm_ntasks.isdigit() and int(slurm_ntasks) > 0 else 1

    BASE_PATH = '/cosma8/data/dp317/dc-naza3/homogeneous/et'
    SNAPBASE  = 'snap_'
    SNAP_NUMS  = [1,15,18]#[1,3,10]

    # Panel order: left (0), right (1), bottom (2)
    RUNS = [

        {'path': os.path.join(BASE_PATH, '5',  ''), 'label': r'$N_\mathrm{H} \approx 10^{23}\ \mathrm{cm}^{-2}$'},
        {'path': os.path.join(BASE_PATH, '5',   ''), 'label': r'$N_\mathrm{H} \approx 10^{22}\ \mathrm{cm}^{-2}$'},
        {'path': os.path.join(BASE_PATH, '5', ''), 'label': r'$N_\mathrm{H} \approx 10^{21}\ \mathrm{cm}^{-2}$'},

    ]

    ## Plotting options
    var        = 'crendens'
    weighted   = 'rho'
    cmap       = 'cmr.amethyst'
    logplot    = True
    ranges     = [1e-2,1e1]#None          # e.g. [1e4, 1e8] to lock all panels to the same scale
    image_proj = 'side'
    proj_on    = False
    proj_fact  = 0.1
    res        = 512
    plotsize   = 1.5          # kpc — same region shown in all three panels
    add_vec    = False

    ## Square figure.  The plot_rect defines the square data area in figure coords.
    ## All three panels overlay the same rectangle; clip paths divide it into thirds.
    PLOT_RECT = (0.08, 0.08, 0.80, 0.84)   # (left, bottom, width, height)

    fig = plt.figure(figsize=(8, 8))

    axes, mappables, snap_times = [], [], []

    for loc, run in enumerate(RUNS):
        print(f'\n--- Loading run: {run["path"]} ---')
        s = load_snap_data(SNAP_NUMS[loc], snappath=run['path'], snapbase=SNAPBASE)
        snap_times.append(calc_snap_time(s))

        ax, mappable = plot_tri_axis(
            s, fig, loc,
            plot_rect  = PLOT_RECT,
            var        = var,
            weighted   = weighted,
            ranges     = ranges,
            cmap       = cmap,
            logplot    = logplot,
            image_proj = image_proj,
            proj_on    = proj_on,
            proj_fact  = proj_fact,
            res        = res,
            plotsize   = plotsize,
            colorbar   = False,
            add_vec    = add_vec,
            numthreads = numthreads,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)
        mappables.append(mappable)

        r_shock, r_reverse = find_shock_radius(s, r_range=(1e-3,plotsize), nbins=500)
        revshock = True

        # Each axis covers the full PLOT_RECT; we must clip circles to this
        # panel's wedge region (same vertices used inside plot_tri_axis).
        from matplotlib.patches import Polygon as MplPolygon
        tri_clip_verts = {
            0: [[0, 0], [0, 1], [0.5, 1], [0.5, 0.5]],
            1: [[0.5, 1], [1, 1], [1, 0], [0.5, 0.5]],
            2: [[0, 0], [0.5, 0.5], [1, 0]],
        }[loc]
        tri_clip = MplPolygon(tri_clip_verts, transform=ax.transAxes, closed=True)
        ax.add_patch(tri_clip)
        tri_clip.set_visible(False)

        cx, cy = s.header['BoxSize'] / 2., s.header['BoxSize'] / 2.
        fwd = plt.Circle((cx, cy), r_shock, color='yellow', fill=False, linestyle='--', linewidth=1.5, label='Forward Shock')
        ax.add_patch(fwd)
        fwd.set_clip_path(tri_clip)

        if revshock:
            rev = plt.Circle((cx, cy), r_reverse, color='cyan', fill=False, linestyle='--', linewidth=1.5, label='Reverse Shock')
            ax.add_patch(rev)
            rev.set_clip_path(tri_clip)

    


    ## Draw the three inverted-Y dividing lines in figure coordinates
    #   ⅄  =  vertical stem (top-centre → centre)
    #        + left  diagonal (centre → bottom-left)
    #        + right diagonal (centre → bottom-right)
    L, B, W, H = PLOT_RECT
    xc = L + W / 2   # horizontal centre
    yc = B + H / 2   # vertical centre (Y fork point)
    yt = B + H        # top of plot area
    xr = L + W        # right edge

    for (xa, ya, xb, yb) in [
        (xc, yt, xc, yc),   # vertical stem
        (xc, yc, L,  B),    # left  diagonal → bottom-left corner
        (xc, yc, xr, B),    # right diagonal → bottom-right corner
    ]:
        fig.add_artist(Line2D([xa, xb], [ya, yb],
                              transform=fig.transFigure,
                              color='black', lw=0.8, zorder=30, clip_on=False))

    ## Single vertical colorbar to the right of the plot area
    cb_gap = 0.015
    cb_w   = 0.025
    cax = fig.add_axes([xr + cb_gap, B, cb_w, H])
    cbar = fig.colorbar(mappables[0], cax=cax, orientation='vertical')
    cbar.set_label(r'$\epsilon_{\rm CR}$', fontsize=40)

    ## Panel labels inside the image
    pad = 0.012
    label_kw = dict(transform=fig.transFigure, fontsize=20, color='white',
                    bbox=dict(boxstyle='round,pad=0.15', fc='none', ec='none'))
    fig.text(L  + pad, yt - pad, RUNS[0]['label'], ha='left',   va='top',    **label_kw)
    fig.text(xr - pad, yt - pad, RUNS[1]['label'], ha='right',  va='top',    **label_kw)
    fig.text(xc,       B  + pad, RUNS[2]['label'], ha='center', va='bottom', **label_kw)

    snap_time_str = f'{snap_times[0]:.3f}' if snap_times else '?'
    # fig.suptitle(
    #     f'Time: {snap_time_str} Myr \n'
    #     r'CR energy density' if weighted else '',
    #     fontsize=12, y=0.975,
    # )

    outdir  = '/cosma8/data/dp317/dc-naza3/snap-plotting/tests/tri_plots'
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f'tri_{var}_snap{number_string(SNAP_NUMS[0])}_{image_proj}.png')
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f'\nSaved to {outfile}')
