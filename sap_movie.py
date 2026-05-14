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

OUTDIR = Path('/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/snap-plotting/tests/beta/anim/')

v = 'nH_cm'

FRAMES_DIR = OUTDIR / f'{v}_frames'
MOVIE_MP4 = OUTDIR / f'{v}_comp_{SNAP_START:03d}_{SNAP_END:03d}.mp4'
MOVIE_GIF = OUTDIR / f'{v}_comp_{SNAP_START:03d}_{SNAP_END:03d}.gif'

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

   ## Set global figure options

    image_proj = 'side'      # side or top viewing angle; top view shows azimuthal curvature
    plotsize =  0.5  # Size in kpc of one panel
    proj_on = False        # Whether to do a slice or a projection
    proj_fact = 0.1         # Fraction of plotsize to project through
    res = 1024               # Pixels per panel

    ## Load the snapshot
    c = 'jet'
    r = [1e-2,5e1]
    add_vec = False
    vec_val = 'vel'
    logplot = True

    norm = False
    cdis = False
    def norm_by_snap0(norm):
        s0 = load_snap_data(0,snappath=output_path,snapbase=SNAPBASE)
        if norm:
            for data_key in ['rho', 'nH_cm', 'pres', 'cren', 'speed','temp', 'crpres','bflden', 'bfldpres', 'wind']:#
                # Avoid division by zero and cast to float to avoid UFuncTypeError
                div = np.median(s0.data[data_key]) if s0.data[data_key].mean() != 0 else 1e-10
                if data_key == 'cren': div = np.median(s0.data['u'])  # Normalize CR energy by initial internal energy, not CR energy:
                if data_key == 'crpres': div = np.median(s0.data['pres'])  # Normalize CR pressure by initial thermal pressure, not CR pressure
                if data_key == 'speed': div = 1e5/unit_v # speed in kms
                s.data[data_key] = s.data[data_key].astype(float) / div
                print(f'Normalized {data_key} by dividing by max value from snap 0: {np.median(s0.data[data_key])}')
        if cdis: 
            s.data[v] *= (s.data['wind']>=0.5)


    ## Plot each axis quadrent
    # Top left

    # Here we're passing the same snap each time, but you could give each one a different snapshot to make e.g. a time series image
    output_path = BASE_PATH + '/new/output_cnocr/'

    s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)
    snap_time = calc_snap_time(s)

    unit_v = s.header['UnitVelocity_in_cm_per_s']
    unit_l = s.header['UnitLength_in_cm'] 
    unit_m = s.header['UnitMass_in_g']
    unit_t = unit_l / unit_v
    unit_rho = unit_m / unit_l**3
    
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
        logplot = logplot,
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

    output_path = BASE_PATH + '/new/output_cbf/'

    s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)
    snap_time = calc_snap_time(s)
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
        logplot = logplot,
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
    output_path = BASE_PATH + '/new/output_cbcr/'

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
        logplot = logplot,
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
    # output_path = BASE_PATH + '/output_cturb/'

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
        logplot = logplot,
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
    fig.colorbar(map_TL, cax=cax_TL, orientation='horizontal', label=r'hydro', ticklocation='top')
    
    cax_BL = fig.add_axes([quad_BL.get_position().x0, quad_BL.get_position().y0 - 0.08, 
                           quad_BL.get_position().width, 0.02])
    fig.colorbar(map_BL, cax=cax_BL, orientation='horizontal', label=r'hydro+B')
    
    cax_TR = fig.add_axes([quad_TR.get_position().x0, quad_TR.get_position().y1 + 0.02, 
                           quad_TR.get_position().width, 0.02])
    fig.colorbar(map_TR, cax=cax_TR, orientation='horizontal', label=r'hydro+B+cr', ticklocation='top')
    
    cax_BR = fig.add_axes([quad_BR.get_position().x0, quad_BR.get_position().y0 - 0.08, 
                           quad_BR.get_position().width, 0.02])
    fig.colorbar(map_BR, cax=cax_BR, orientation='horizontal', label=r'hydro+B+cr')

    fig.suptitle(f'Snapshot {snap_num:03d} — Time: {snap_time:.1f} Myr \n$n_\\mathrm{{H}}$ [cm$^{-3}$]', fontsize=12, y=1.002)

    #     # Add colorbars - top row at the top, bottom row at the bottom
    # cax_TL = fig.add_axes([quad_TL.get_position().x0, quad_TL.get_position().y1, 
    #                        quad_TL.get_position().width, 0.02])
    # fig.colorbar(map_TL, cax=cax_TL, orientation='horizontal', ticklocation='top')#, label=r'$T/T_0$')
    
    # cax_BL = fig.add_axes([quad_BL.get_position().x0, quad_BL.get_position().y0 - 0.02, 
    #                        quad_BL.get_position().width, 0.02])
    # fig.colorbar(map_BL, cax=cax_BL, orientation='horizontal')#, label=r'$n_\mathrm{H}/n_0$')
    
    # cax_TR = fig.add_axes([quad_TR.get_position().x0, quad_TR.get_position().y1, 
    #                        quad_TR.get_position().width, 0.02])
    # cb_TR = fig.colorbar(map_TR, cax=cax_TR, orientation='horizontal', ticklocation='top')#, label=r'$E_\mathrm{CR}/E_0$')
    # # cb_TR.set_ticks(cb_TR.get_ticks()[1:])
    
    # cax_BR = fig.add_axes([quad_BR.get_position().x0, quad_BR.get_position().y0 - 0.02, 
    #                        quad_BR.get_position().width, 0.02])
    
    # cb_BR = fig.colorbar(map_BR, cax=cax_BR, orientation='horizontal')#, label=r'$X_\mathrm{CR}=P_\mathrm{CR}/P_{th}$')
    # cb_BR.set_ticks(cb_BR.get_ticks()[1:])

    # quad_TL.text(0.05, 0.95, r'$B=0 \; \mathrm{G}$+no diff', transform=quad_TL.transAxes, color='maroon', ha='left', va='top', weight='bold',fontsize=15)
    # quad_BL.text(0.05, 0.05, r'$B_x = 10^{-6} \; \mathrm{G}$', transform=quad_BL.transAxes, color='maroon', ha='left', va='bottom', weight='bold',fontsize=15)
    # quad_TR.text(0.95, 0.95, r'$B_\mathrm{turb} = 10^{-6} \; \mathrm{G}$', transform=quad_TR.transAxes, color='maroon', ha='right', va='top', weight='bold',fontsize=15)
    # quad_BR.text(0.95, 0.05, r'$B_{\phi} = 10^{-6} \; \mathrm{G}$', transform=quad_BR.transAxes, color='maroon', ha='right', va='bottom', weight='bold',fontsize=15)

    # r_cool, v_sh_kms = find_Rcool(snappath=BASE_PATH + '/output_cool/', snapnum=snap_num, L_AGN=1e45)
    # for ax in [quad_TL, quad_TR, quad_BL, quad_BR]:
    #     ax.add_patch(plt.Circle((s.header['BoxSize']/2., s.header['BoxSize']/2.), r_cool, color='magenta', fill=False, linestyle='--', linewidth=1.5, label='Cooling Radius'))

    frame_path = FRAMES_DIR / f'{v}_{snap_num:03d}.png'
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
    frame_paths_expected = [FRAMES_DIR / f'{v}_{snapnum:03d}.png' for snapnum in snapnums]
    pending = [snapnum for snapnum, frame_path in zip(snapnums, frame_paths_expected) if not frame_path.exists()]

    worker_args = [(snapnum, rho_0) for snapnum in pending]
    slurm_ntasks = os.getenv('SLURM_NTASKS', '').strip()
    nproc = int(slurm_ntasks) if slurm_ntasks.isdigit() and int(slurm_ntasks) > 0 else 1
    # nproc = min(8, len(pending), max(1, multiprocessing.cpu_count() - 1))
    if pending:
        print(f"Preparing {len(pending)} snapshots with {nproc} processes...")
        with ProcessPoolExecutor(max_workers=nproc) as pool:
            for _ in pool.map(render_frame_from_data, worker_args):
                pass
    else:
        print("All frame PNGs already exist; skipping frame rendering.")


    frame_paths = [str(frame_path) for frame_path in frame_paths_expected if frame_path.exists()]
    if len(frame_paths) != len(snapnums):
        missing = [f'{v}_{snapnum:03d}.png' for snapnum, frame_path in zip(snapnums, frame_paths_expected) if not frame_path.exists()]
        raise RuntimeError(f"Missing frames after generation: {missing}")

    build_movie(frame_paths, fps=3)


if __name__ == '__main__':
    main()