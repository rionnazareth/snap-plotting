from lib import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np
import scienceplots
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing
import contextlib
import os

plt.style.use(['science'])

SNAPBASE = 'snap_'
SNAP_START = 1
SNAP_END = 10
RAD_WIND = 0.0078125
MAX_POINTS = 500000

SNAPPATH_CR = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/old/output_cr600/'
SNAPPATH_NOCR = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/old/output_homo/'

OUTDIR = Path('/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/snap-plotting/tests/pap')
FRAMES_DIR = OUTDIR / 'tempvsrho_frames'
MOVIE_MP4 = OUTDIR / f'tempvsrho_comp_{SNAP_START:03d}_{SNAP_END:03d}.mp4'
MOVIE_GIF = OUTDIR / f'tempvsrho_comp_{SNAP_START:03d}_{SNAP_END:03d}.gif'

def get_shell(s, r_range=(1e-3, 1)):
    r, rho = radial_profile_log(s, 'rho', r_range=r_range, nbins=500)
    
    idx = np.nanargmax(rho)
    grad = np.abs(np.gradient(rho, r))
    tol=1e-3
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
    
    return r, left, right

def prepare_snapshot_data(args):
    snapnum, rho_0 = args
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        s_cr = load_snap_data(snapnum, snappath=SNAPPATH_CR, snapbase=SNAPBASE)
        s = load_snap_data(snapnum, snappath=SNAPPATH_NOCR, snapbase=SNAPBASE)

    rho_cr = s_cr.data['rho'] / rho_0
    temp_cr = s_cr.data['temp']
    rho_nocr = s.data['rho'] / rho_0
    temp_nocr = s.data['temp']

    mask = np.ones_like(s.data['r'], dtype=bool)
    mask_cr = np.ones_like(s_cr.data['r'], dtype=bool)
    mask &= (s.data['r'] >= RAD_WIND)
    mask_cr &= (s_cr.data['r'] >= RAD_WIND)

    # mask &=  (s.data['mach'] == 0) 
    # mask_cr &= (s_cr.data['mach'] == 0) 

    # mask &= (s.data['r'] >= shell_start) 
    # mask_cr &= (s_cr.data['r'] >= shell_cr_start) 

    # mask &= (s.data['r'] <= shell_end) 
    # mask_cr &= (s_cr.data['r'] <= shell_cr_end) 

    # mask &=  (s.data['wind'] > 1e-5)
    # mask_cr&= (s_cr.data['wind'] > 1e-5)  

    rho_cr_sel = rho_cr[mask_cr]
    temp_cr_sel = temp_cr[mask_cr]
    ratio_cr_sel = s_cr.data['cren'][mask_cr] / s_cr.data['u'][mask_cr]
    rho_nocr_sel = rho_nocr[mask]
    temp_nocr_sel = temp_nocr[mask]

    # rng = np.random.default_rng(snapnum)
    # if rho_cr_sel.size > MAX_POINTS:
    #     idx_cr = rng.choice(rho_cr_sel.size, size=MAX_POINTS, replace=False)
    #     rho_cr_sel = rho_cr_sel[idx_cr]
    #     temp_cr_sel = temp_cr_sel[idx_cr]
    #     ratio_cr_sel = ratio_cr_sel[idx_cr]

    # if rho_nocr_sel.size > MAX_POINTS:
    #     idx_nocr = rng.choice(rho_nocr_sel.size, size=MAX_POINTS, replace=False)
    #     rho_nocr_sel = rho_nocr_sel[idx_nocr]
    #     temp_nocr_sel = temp_nocr_sel[idx_nocr]

    return {
        'snapnum': snapnum,
        'rho_cr': rho_cr_sel,
        'temp_cr': temp_cr_sel,
        'ratio_cr': ratio_cr_sel,
        'rho_nocr': rho_nocr_sel,
        'temp_nocr': temp_nocr_sel,
    }


def render_frame_from_data(frame_data):
    snapnum = frame_data['snapnum']
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        s_cr = load_snap_data(snapnum, snappath=SNAPPATH_CR, snapbase=SNAPBASE)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    sc_cr = ax[0].scatter(
        frame_data['rho_cr'],
        frame_data['temp_cr'],
        s=0.1,
        c=frame_data['ratio_cr'],
        cmap='vanimo',
        alpha=0.1,
        norm=LogNorm(vmin=1e-4, vmax=1e2),
    )
    cbar = fig.colorbar(sc_cr, ax=ax[0])
    cbar.set_label(r'$E_{\rm CR}/E_{\rm th}$', fontsize=11)
    cbar.solids.set_alpha(1)

    # ax[0].scatter(frame_data['rho_nocr'], frame_data['temp_nocr'], s=0.1, color="#4CD4D4", alpha=0.01)
    # ax[0].scatter([], [], s=20, color="#4CD4D4", label='Without CR', alpha=1.0)

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlim((1e-3, 30))
    ax[0].set_ylim((1e1, 4e9))
    ax[0].set_xlabel(r'$\rho / \rho_0$', fontsize=14)
    ax[0].set_ylabel(r'$T [K]$', fontsize=14)
    ax[0].set_title(f'Snapshot {snapnum:03d}')
    # ax[0].legend(fontsize=10, loc='lower left')

    center = [0.5, 0.5, 0.5]#[s_cr.header['BoxSize'] / 2] * 3
    box_zoom = [1, 1]
    proj_on = False
    proj_fact = 0.1
    s_cr.axplot_Aweightedslice(ax[1],
                        value='ecth', weights='rho', cmap='vanimo', colorbar=False, logplot=True, vrange=(1e-4, 1e2),
                        center=center, box=box_zoom, res=1024,
                        proj=proj_on, proj_fact=proj_fact, axes=[0,2]
                    )
    ax[1].set_xticklabels(np.round(ax[1].get_xticks()-s_cr.boxsize/2, decimals=2))
    ax[1].set_yticklabels(np.round(ax[1].get_yticks()-s_cr.boxsize/2, decimals=2))

    fig.tight_layout()

    frame_path = FRAMES_DIR / f'tempvsrho_{snapnum:03d}.png'
    fig.savefig(frame_path, dpi=150)
    plt.close(fig)
    return str(frame_path)


def build_movie(frame_paths, fps=3):
    frame_paths = sorted(frame_paths)
    first_img = plt.imread(frame_paths[0])

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    im = ax.imshow(first_img)
    ax.axis('off')
    fig.tight_layout(pad=0)

    def update(i):
        im.set_data(plt.imread(frame_paths[i]))
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(frame_paths), interval=1000 / fps, blit=True)

    if writers.is_available('ffmpeg'):
        anim.save(MOVIE_MP4, writer='ffmpeg', fps=fps)
        print(f"Saved movie: {MOVIE_MP4}")
    else:
        print("MovieWriter ffmpeg unavailable; saving GIF with Pillow.")
        anim.save(MOVIE_GIF, writer='pillow', fps=fps)
        print(f"Saved movie: {MOVIE_GIF}")

    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        s0 = load_snap_data(0, snappath=SNAPPATH_NOCR, snapbase=SNAPBASE)
    rho_0 = s0.data['rho'].mean()

    snapnums = list(range(SNAP_START, SNAP_END + 1))
    frame_paths_expected = [FRAMES_DIR / f'tempvsrho_{snapnum:03d}.png' for snapnum in snapnums]
    pending = [snapnum for snapnum, frame_path in zip(snapnums, frame_paths_expected) if not frame_path.exists()]

    worker_args = [(snapnum, rho_0) for snapnum in pending]
    nproc = 76#min(8, len(pending), max(1, multiprocessing.cpu_count() - 1))
    if pending:
        print(f"Preparing {len(pending)} snapshots with {nproc} processes...")

        with ProcessPoolExecutor(max_workers=nproc) as executor:
            for frame_data in executor.map(prepare_snapshot_data, worker_args):
                render_frame_from_data(frame_data)

    frame_paths = [str(frame_path) for frame_path in frame_paths_expected if frame_path.exists()]
    if len(frame_paths) != len(snapnums):
        missing = [f'tempvsrho_{snapnum:03d}.png' for snapnum, frame_path in zip(snapnums, frame_paths_expected) if not frame_path.exists()]
        raise RuntimeError(f"Missing frames after generation: {missing}")

    build_movie(frame_paths, fps=3)


if __name__ == '__main__':
    main()