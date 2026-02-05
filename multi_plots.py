import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import arepo_run as arun

# plt.rcParams.update({
#     "text.usetex": True,         # use LaTeX for all text
#     "font.family": "serif",      # choose serif font
#     "font.serif": ["Computer Modern Roman"],  # standard LaTeX font
# })

# --- helper to process 1 snapshot in parallel ---
def process_snapshot(args):
    i, value, axes_plot, center, box, res, proj, proj_fact, vrange, cmap, snap_path = args
    
    o  = arun.Run(snappath=snap_path,
                  snapbase="snap_")

    s  = o.loadSnap(snapnum=i)

    gamma = 5./3
    unit_v = 1e5

    # ---- derived quantities ----
    if value == 'temp':
        kB = 1.381e-16
        mP = 1.66e-24
        xH = 0.76
        meanMolecularWeight = 0.6*mP #4*mP / (1 + 3*xH + 4*xH * s.data['ne'])
        s.data['temp'] = (gamma - 1) * meanMolecularWeight / kB * s.data['u'] * unit_v**2

    if value == 'speed':
        s.data['speed'] = np.linalg.norm(s.data['vel'], axis=1)

    if value == 'vortmag':
        s.data['vortmag'] = np.linalg.norm(s.data['vortmag'], axis=1)

    if value == 'grar_rho':
        s.data['grar_rho'] = np.linalg.norm(s.data['grar'], axis=1) / s.data['rho']
    
    if value == 'energdens':
        s.data['energdens'] = s.data['u']*s.data['rho']

    # ---- render slice to an offscreen figure ----
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)

    s.plot_Aslice(
        value=value, axes=axes_plot, cmap=cmap, colorbar=True,
        center=center, box=box, res=res,
        logplot=True, vrange=vrange, minimum=1e-10, newfig=False, proj=proj, proj_fact=proj_fact
    )
    ax.set_xlabel(r'$x\,[{\rm kpc}]$')
    ax.set_ylabel(r'$z\,[{\rm kpc}]$')
    ax.set_title(f"snap {i}")

    # convert figure → numpy array to send back to main process
    fig.canvas.draw()

    # Get RGBA buffer (new API)
    buf = fig.canvas.buffer_rgba()

    # Convert to NumPy array
    w, h = fig.canvas.get_width_height()
    img = np.asarray(buf, dtype=np.uint8).reshape((h, w, 4))

    # Remove alpha channel
    img = img[:, :, :3]

    plt.close(fig)
    return img

def plot_multiple(value, num_proc=4, num_snaps=10, snap_offset=0, save_path='',snap_path='', axes_plot=[0,2], 
                  vrange=False, center=[500,500,500], box=[1000,1000], 
                  res=1024, proj=False, proj_fact=0.5, cmap='gnuplot'):

    if num_snaps==0:
        ncols = 1
        nrows = 1
    else:
        ncols = 3
        nrows = int(np.ceil((num_snaps+0.01) / ncols))# +0.01 to offset when numsnaps = 3n

    # Prepare argument list for all snapshots
    args_list = [
        (i, value, axes_plot, center, box, res, proj, proj_fact, vrange, cmap, snap_path)
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

    fig.suptitle(value.capitalize(), fontsize=16)
    plt.tight_layout()

    outfile = os.path.join(save_path, f"multi_{value}.png")
    plt.savefig(outfile, dpi=600, bbox_inches="tight")

plot_multiple('temp', num_proc=32, save_path='/cosma8/data/dp317/dc-naza3/gasCloudNfw/plotting/plots', 
              snap_path='/cosma8/data/dp317/dc-naza3/gasCloudNfw/output_refined',
               num_snaps=3, snap_offset=0, axes_plot=[0, 2], box=[100,100], proj=False, vrange=(1e5,1e9), cmap='gnuplot')

#dict_keys(['pos', 'rho', 'grar', 'u', 'mass', 'id', 'pres', 'vel', 'vol', 'vort', 'type']) 
#if 'ne' is present, can add temperature with 'temp'
#also added 'speed', 'vortmag', 'grar_rho' (\grad rho/rho),