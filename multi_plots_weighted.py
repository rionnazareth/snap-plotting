import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import arepo_run as arun
from tests.lib import *

# plt.rcParams.update({
#     "text.usetex": True,         # use LaTeX for all text
#     "font.family": "serif",      # choose serif font
#     "font.serif": ["Computer Modern Roman"],  # standard LaTeX font
# })

# --- helper to process 1 snapshot in parallel ---
def process_snapshot(args):
    i, value, axes_plot, box, res, proj, proj_fact, vrange, cmap, snap_path, weighted, weights, num_proc = args

    s  = load_snap_data(i, snappath=snap_path, snapbase="snap_")

    unit_v = s.header['UnitVelocity_in_cm_per_s']
    unit_l = s.header['UnitLength_in_cm'] 
    unit_m = s.header['UnitMass_in_g']
    unit_t = unit_l / unit_v
    unit_rho = unit_m / unit_l**3

    norm = False
    s0 = load_snap_data(0,snappath=snap_path,snapbase="snap_")
    if norm:
        for data_key in ['rho', 'nH_cm', 'pres',  'speed','temp', 'wind','vrad']:#
            # Avoid division by zero and cast to float to avoid UFuncTypeError
            div = np.median(s0.data[data_key]) if s0.data[data_key].mean() != 0 else 1e-10
            if data_key == 'cren': div = np.median(s0.data['u'])  # Normalize CR energy by initial internal energy, not CR energy:
            if data_key == 'crpres': div = np.median(s0.data['pres'])  # Normalize CR pressure by initial thermal pressure, not CR pressure
            if data_key == 'speed' or data_key == 'vrad': div = 1e5/unit_v # speed in kms
            s.data[data_key] = s.data[data_key].astype(float) / div
            print(f'Normalized {data_key} by dividing by max value from snap 0: {np.median(s0.data[data_key])}')
    
    center = s.header['BoxSize'] / 2.0
    print(center)

    # ---- render slice to an offscreen figure ----
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)

    # Use weighted or unweighted plot
    if weighted and weights:
        s.plot_Aweightedslice(
            value=value, weights=weights, cmap=cmap, colorbar=True,
            center=center, box=box, res=res,
            logplot=True, vrange=vrange, minimum=1e-10, 
            newfig=False, proj=proj, proj_fact=proj_fact, numthreads=num_proc
        )
    else:
        s.plot_Aslice(
            value=value, axes=axes_plot, cmap=cmap, colorbar=True,
            center=center, box=box, res=res,
            logplot=True, vrange=vrange, minimum=1e-10, newfig=False, proj=proj, proj_fact=proj_fact, numthreads=num_proc
        )

    # Customize axes
    ax = plt.gca()
    ax.set_xlabel(r'$x\,[{\rm kpc}]$')
    ax.set_ylabel(r'$z\,[{\rm kpc}]$')
    ax.set_title(f"snap {i}")

    # convert figure → numpy array to send back to main process
    fig = plt.gcf()
    fig.canvas.draw()

    # Get RGBA buffer (new API)
    buf = fig.canvas.buffer_rgba()

    # Convert to NumPy array
    w, h = fig.canvas.get_width_height()
    img = np.asarray(buf, dtype=np.uint8).reshape((h, w, 4))

    # Remove alpha channel
    img = img[:, :, :3]

    plt.close('all')
    return img

def plot_multiple(value, num_proc=4, num_snaps=10, snap_offset=0, save_path='',snap_path='', axes_plot=[0,2], 
                  vrange=False, box=[1000,1000], 
                  res=1024, proj=False, proj_fact=0.5, cmap='gnuplot', weighted=False, weights='rho'):
    """
    Create a multi-panel figure from multiple snapshots.
    
    Parameters
    ----------
    value : str
        Quantity to plot ('temp', 'rho', 'u', 'speed', 'vortmag', 'grar_rho', 'energdens', etc.)
    num_proc : int
        Number of parallel processes
    num_snaps : int
        Number of snapshots to plot
    snap_offset : int
        First snapshot number
    save_path : str
        Directory to save output
    snap_path : str
        Path to snapshot directory
    axes_plot : list
        Axes to plot [x, z]
    vrange : tuple
        Color range (min, max)
    center : list
        Center position [x, y, z]
    box : list
        Size of region [width, height]
    res : int
        Resolution
    proj : bool
        Use projection instead of slice
    proj_fact : float
        Projection depth factor
    cmap : str
        Colormap
    weighted : bool
        Use density-weighted plot (default: False)
    weights : str
        Field to weight by - 'rho', 'mass', 'vol' (default: 'rho')
    """

    if num_snaps==0:
        ncols = 1
        nrows = 1
    else:
        ncols = 3
        nrows = int(np.ceil((num_snaps+0.01) / ncols))# +0.01 to offset when numsnaps = 3n

    # Prepare argument list for all snapshots
    args_list = [
        (i, value, axes_plot, box, res, proj, proj_fact, vrange, cmap, snap_path, weighted, weights, num_proc)
        for i in range(snap_offset, snap_offset+num_snaps+1)
    ]

    # --- run in parallel ---
    with Pool(processes=num_proc) as pool:
        images = pool.map(process_snapshot, args_list)

    # --- assemble into one big multi-panel figure ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if num_snaps==0: 
        axes = [axes]
    else:    
        axes = axes.flatten()

    for ax, img, i in zip(axes, images, range(len(images))):
        ax.imshow(img)
        ax.set_axis_off()

    # Hide unused axes
    for ax in axes[len(images):]:
        ax.set_visible(False)

    title = value.capitalize()
    if weighted:
        title += f" ({weights}-weighted)"
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    outfile = os.path.join(save_path, f"multi_{value}.png")
    plt.savefig(outfile, dpi=600, bbox_inches="tight")

save_path = '/cosma8/data/dp317/dc-naza3/snap-plotting/tests/rhov'
snap_path = '/cosma8/data/dp317/dc-naza3/homogeneous/rho_vary/0.5'


# Example usage - unweighted (original)
slurm_ntasks = os.getenv('SLURM_NTASKS', '').strip()
numthreads = int(slurm_ntasks) if slurm_ntasks.isdigit() and int(slurm_ntasks) > 0 else 1
plot_multiple('nH_cm', 
              num_proc=numthreads, save_path=save_path, 
              snap_path=snap_path,
               num_snaps=13, snap_offset=0, axes_plot=[0, 2], box=[1.5, 1.5], proj=False, proj_fact=0.3, 
               cmap='jet', weighted=True, weights='rho',vrange=None)

print(f"Plots saved in {save_path}")
# Example usage - density-weighted
# plot_multiple('temp', num_proc=32, save_path='/cosma8/data/dp317/dc-naza3/gasCloudNfw/plotting/plots', 
#               snap_path='/cosma8/data/dp317/dc-naza3/gasCloudNfw/output_refined',
#                num_snaps=3, snap_offset=0, axes_plot=[0, 2], box=[100,100], proj=True, proj_fact=0.25, 
#                vrange=(1e5,1e9), cmap='gnuplot', weighted=True, weights='rho')

#dict_keys(['pos', 'rho', 'grar', 'u', 'mass', 'id', 'pres', 'vel', 'vol', 'vort', 'type']) 
#if 'ne' is present, can add temperature with 'temp'
#also added 'speed', 'vortmag', 'grar_rho' (\grad rho/rho),