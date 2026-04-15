from tests.lib import *
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

OUTDIR = Path('/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/snap-plotting/tests/quad')
FRAMES_DIR = OUTDIR / 'quad_frames'
MOVIE_MP4 = OUTDIR / f'quad_comp_{SNAP_START:03d}_{SNAP_END:03d}.mp4'
MOVIE_GIF = OUTDIR / f'quad_comp_{SNAP_START:03d}_{SNAP_END:03d}.gif'

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


def render_frame_from_data(frame_data):
    snap_num, _ = frame_data
    print(f'Rendering snapshot {snap_num:03d}...')
    slurm_ntasks = os.getenv('SLURM_NTASKS', '').strip()
    numthreads = int(slurm_ntasks) if slurm_ntasks.isdigit() and int(slurm_ntasks) > 0 else 1
    BASE_PATH = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion'
    SNAPBASE = 'snap_'
    SNAPFILETYPE = '.hdf5'

    GAMMA = 5/3 # heat capaticy ratio for monatomic gas
    HYDROGENMASS_FRAC = 0.76

    # k_B = const.k_B.to((u.Msun*1e10)*(u.km/u.s)**2/u.K).value
    # m_p = const.m_p.to(u.Msun*1e10).value

    k_B       = 1.381e-16
    m_p       = 1.66e-24



    ## Set up figure & axis grid
    fig = plt.figure(figsize=(8,8))

    outer = gridspec.GridSpec(1, 1, wspace=0.2)
    quad_subs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], hspace=0, wspace=0)





    ## Set global figure options

    image_proj = 'side'     # side or top viewing angle
    plotsize =  0.55   # Size in kpc of one panel
    proj_on = False        # Whether to do a slice or a projection
    proj_fact = 0.1         # Fraction of plotsize to project through
    res = 1024               # Pixels per panel

    ## Load the snapshot
    v = 'rho'
    c = 'gnuplot'
    r = [1e-3,1e1]
    output_path = BASE_PATH + '/old/output_cr600/'

    s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)
    snap_time = calc_snap_time(s)
    add_vec = False

    norm = True
    def norm_by_snap0(norm):
        s0 = load_snap_data(0,snappath=output_path,snapbase=SNAPBASE)
        if norm:
            for data_key in ['rho', 'nH_cm', 'pres']:
                # Avoid division by zero and cast to float to avoid UFuncTypeError
                div = s0.data[data_key].mean() if s0.data[data_key].mean() != 0 else 1e-10
                if data_key == 'cren': div = s0.data['u'].mean()  # Normalize CR energy by initial internal energy, not CR energy:
                s.data[data_key] = s.data[data_key].astype(float) / div
                print(f'Normalized {data_key} using snap 0 mean: {s0.data[data_key].mean()}')
    ## Plot each axis quadrent
    # Top left

    # Here we're passing the same snap each time, but you could give each one a different snapshot to make e.g. a time series image
    norm_by_snap0(norm)
    quad_TL, map_TL = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [0,0],
        var = v,
        weighted = 'rho', # or None
        ranges = r,
        cmap = c,
        logplot = True,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        numthreads = numthreads,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val='bfld'
        )

    output_path = BASE_PATH + '/old/output_homo/'

    s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)
    snap_time = calc_snap_time(s)
    add_vec = False
    # Bottom left
    norm_by_snap0(norm)
    quad_BL, map_BL = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [1,0],
        var = v,
        weighted = 'rho', # or None
        ranges = r,
        cmap = c,
        logplot = True,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        numthreads = numthreads,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val='bfld'
        )

    # snap_num = 1
    output_path = BASE_PATH + '/output_cool/'

    s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)
    snap_time = calc_snap_time(s)
    # Bottom right
    norm_by_snap0(norm)
    quad_BR, map_BR = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [1,1],
        var = v,
        weighted = 'rho', # or None
        ranges = r,
        cmap = c,
        logplot = True,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        numthreads = numthreads,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val='bfld'
        )

    # Top right
    output_path = BASE_PATH + '/output_cool/'

    s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)
    snap_time = calc_snap_time(s)
    norm_by_snap0(norm)
    quad_TR, map_TR = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [0,1],
        var = v,
        weighted = 'rho', # or None
        ranges = r,
        cmap = c,
        logplot = True,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        numthreads = numthreads,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val='bfld'
        )



    # plt.show()

    # Adjust figure to make room for colorbars
    fig.subplots_adjust(bottom=0.14, top=0.88)

    # Add colorbars - top row at the top, bottom row at the bottom
    cax_TL = fig.add_axes([quad_TL.get_position().x0, quad_TL.get_position().y1 + 0.02, 
                        quad_TL.get_position().width, 0.02])
    fig.colorbar(map_TL, cax=cax_TL, orientation='horizontal', label='no cooling', ticklocation='top')
    
    cax_BL = fig.add_axes([quad_BL.get_position().x0, quad_BL.get_position().y0 - 0.08, 
                        quad_BL.get_position().width, 0.02])
    fig.colorbar(map_BL, cax=cax_BL, orientation='horizontal', label='no cooling + no CRs')
    
    cax_TR = fig.add_axes([quad_TR.get_position().x0, quad_TR.get_position().y1 + 0.02, 
                        quad_TR.get_position().width, 0.02])
    fig.colorbar(map_TR, cax=cax_TR, orientation='horizontal', label='cooling', ticklocation='top')
    
    cax_BR = fig.add_axes([quad_BR.get_position().x0, quad_BR.get_position().y0 - 0.08, 
                        quad_BR.get_position().width, 0.02])
    fig.colorbar(map_BR, cax=cax_BR, orientation='horizontal', label='cooling')
    fig.suptitle(f'Snapshot {snap_num:03d} — Time: {snap_time:.1f} Myr — {v}', fontsize=12, y=1.002)

    frame_path = FRAMES_DIR / f'quad_{snap_num:03d}.png'
    fig.savefig(frame_path, dpi=150)
    plt.close(fig)
    return str(frame_path)


def build_movie(frame_paths, fps=3):
    frame_paths = sorted(frame_paths)
    first_img = plt.imread(frame_paths[0])
    height, width = first_img.shape[:2]
    dpi = 100

    # Match the encoder canvas to the native frame pixel size to avoid blur.
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    im = ax.imshow(first_img, interpolation='none', resample=False)
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def update(i):
        im.set_data(plt.imread(frame_paths[i]))
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(frame_paths), interval=1000 / fps, blit=True)

    if writers.is_available('ffmpeg'):
        anim.save(MOVIE_MP4, writer='ffmpeg', fps=fps, dpi=dpi, bitrate=12000)
        print(f"Saved movie: {MOVIE_MP4}")
    else:
        print("MovieWriter ffmpeg unavailable; saving GIF with Pillow.")
        anim.save(MOVIE_GIF, writer='pillow', fps=fps, dpi=dpi)
        print(f"Saved movie: {MOVIE_GIF}")

    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        s0 = load_snap_data(0, snappath=SNAPPATH_NOCR, snapbase=SNAPBASE)
    rho_0 = s0.data['rho'].mean()

    snapnums = list(range(SNAP_START, SNAP_END + 1))
    frame_paths_expected = [FRAMES_DIR / f'quad_{snapnum:03d}.png' for snapnum in snapnums]
    pending = [snapnum for snapnum, frame_path in zip(snapnums, frame_paths_expected) if not frame_path.exists()]

    worker_args = [(snapnum, rho_0) for snapnum in pending]
    nproc = min(8, len(pending), max(1, multiprocessing.cpu_count() - 1))
    if pending:
        print(f"Preparing {len(pending)} snapshots with {nproc} processes...")
        with ProcessPoolExecutor(max_workers=nproc) as pool:
            for _ in pool.map(render_frame_from_data, worker_args):
                pass
    else:
        print("All frame PNGs already exist; skipping frame rendering.")


    frame_paths = [str(frame_path) for frame_path in frame_paths_expected if frame_path.exists()]
    if len(frame_paths) != len(snapnums):
        missing = [f'quad_{snapnum:03d}.png' for snapnum, frame_path in zip(snapnums, frame_paths_expected) if not frame_path.exists()]
        raise RuntimeError(f"Missing frames after generation: {missing}")

    build_movie(frame_paths, fps=3)


if __name__ == '__main__':
    main()