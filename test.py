"""
make_movie.py - Create movies from simulation snapshots

This script generates movies from AREPO snapshots, showing how a given
quantity (density, internal energy, temperature, etc.) evolves over time.

Available derived quantities:
- 'temp': temperature (computed from internal energy)
- 'speed': velocity magnitude
- 'vortmag': vorticity magnitude
- 'grar_rho': normalized density gradient (grad(rho)/rho)
- 'energdens': energy density (u * rho)

Available built-in quantities:
- 'rho': density
- 'u': internal energy
- 'pres': pressure
- 'vel': velocity
- 'vort': vorticity
- etc. (check your snapshot data)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import arepo_run as arun
from multiprocessing import Pool


def process_snapshot(args):
    """Process a single snapshot and return the rendered frame."""
    i, value, axes_plot, center, box, res, proj, proj_fact, vrange, cmap, logplot, snap_path, weighted, weights = args
    
    o = arun.Run(snappath=snap_path, snapbase="snap_")
    s = o.loadSnap(snapnum=i)
    
    gamma = 5.0 / 3
    unit_v = 1e5
    kB = 1.381e-16
    mP = 1.66e-24
    xH = 0.76
    meanMolecularWeight = 0.6 * mP
    
    # Compute derived quantities if needed
    if value == 'temp':
        s.data['temp'] = (gamma - 1) * meanMolecularWeight / kB * s.data['u'] * unit_v**2
    elif value == 'speed':
        s.data['speed'] = np.linalg.norm(s.data['vel'], axis=1)
    elif value == 'vortmag':
        s.data['vortmag'] = np.linalg.norm(s.data['vort'], axis=1)
    elif value == 'grar_rho':
        s.data['grar_rho'] = np.linalg.norm(s.data['grar'], axis=1) / s.data['rho']
    elif value == 'energdens':
        s.data['energdens'] = s.data['u'] * s.data['rho']
    
    # Create a temporary figure for this frame with black background
    temp_fig = plt.figure(figsize=(8, 7), dpi=100, facecolor='black')
    
    # Use weighted or unweighted plot
    if weighted and weights:
        s.plot_Aweightedslice(
            value=value, weights=weights, cmap=cmap, colorbar=True,
            center=center, box=box, res=res,
            logplot=logplot, vrange=vrange, minimum=1e-10, 
            newfig=True, proj=proj, proj_fact=proj_fact, cblabel=r'$T\;[K]$'
        )
    else:
        s.plot_Aslice(
            value=value, axes=axes_plot, cmap=cmap, colorbar=True,
            center=center, box=box, res=res,
            logplot=logplot, vrange=vrange, minimum=1e-10, 
            newfig=True, proj=proj, proj_fact=proj_fact, cblabel=r'$T\;[K]$'
        )
    
    # Get current axes and customize with black background and white text
    ax = plt.gca()
    ax.set_facecolor('black')
    ax.set_xlabel(r'$x\,[{\rm kpc}]$', color='white')
    ax.set_ylabel(r'$z\,[{\rm kpc}]$', color='white')
    ax.set_title(f"{i*5} Myr", fontsize=12, color='white')
    
    # Make tick labels and ticks white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    # Make spines (axes borders) white
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    
    # Colorbar on dark background
    temp_fig = plt.gcf()
    temp_fig.patch.set_facecolor('black')
    for cax in temp_fig.axes:
        if cax is ax:
            continue  # skip main plot axes
        cax.set_facecolor('black')
        cax.tick_params(axis='both', colors='white')
        if hasattr(cax, 'yaxis'):
            cax.yaxis.label.set_color('white')
        if hasattr(cax, 'xaxis'):
            cax.xaxis.label.set_color('white')
        for spine in cax.spines.values():
            spine.set_edgecolor('white')
    
    # Draw and convert to image array
    temp_fig.canvas.draw()
    buf = temp_fig.canvas.buffer_rgba()
    w, h = temp_fig.canvas.get_width_height()
    img = np.asarray(buf, dtype=np.uint8).reshape((h, w, 4))
    img_rgb = img[:, :, :3]  # Remove alpha channel
    
    plt.close('all')
    
    return img_rgb


def make_movie(value, snap_path, snap_offset=0, num_snaps=10, 
               save_path='', movie_name='movie.gif', fps=10, num_proc=4,
               axes_plot=[0, 2], center=None, box=None, 
               res=512, proj=False, proj_fact=0.5, vrange=None, 
               cmap='gnuplot', logplot=True, weighted=False, weights='rho'):
    """
    Create a movie from snapshots showing evolution of a quantity.
    
    Parameters
    ----------
    value : str
        The quantity to plot ('rho', 'u', 'temp', 'speed', 'vortmag', 
        'grar_rho', 'energdens', etc.)
    snap_path : str
        Path to directory containing snapshots
    snap_offset : int, optional
        First snapshot number (default: 0)
    num_snaps : int, optional
        Number of snapshots to include (default: 10)
    save_path : str, optional
        Directory to save movie (if empty, displays in window)
    movie_name : str, optional
        Name of output movie file with extension (default: 'movie.gif')
        Supported: .gif, .mp4 (requires ffmpeg)
    fps : int, optional
        Frames per second (default: 10)
    num_proc : int, optional
        Number of parallel processes to use (default: 4)
    axes_plot : list, optional
        Which axes to plot [x, z] (default: [0, 2] for x-z plane)
    center : list, optional
        Center position [x, y, z] for slice (default: center of domain)
    box : list, optional
        Size of slice [width, height] in kpc (default: [1000, 1000])
    res : int, optional
        Resolution of slice (default: 512)
    proj : bool, optional
        Use projection instead of slice (default: False)
    proj_fact : float, optional
        Projection factor (default: 0.5)
    vrange : tuple, optional
        Color range (min, max) - if None, auto-scaled for each frame
    cmap : str, optional
        Colormap to use (default: 'gnuplot')
    logplot : bool, optional
        Use log scale for values (default: True)
    weighted : bool, optional
        Use density-weighted plot (default: False)
    weights : str, optional
        Field to use for weighting (default: 'rho' for density-weighted)
        Common choices: 'rho', 'mass', 'vol'
    
    Returns
    -------
    None
        Saves movie to file or displays in window
    """
    
    print(f"\nGenerating movie for '{value}'...")
    if weighted:
        print(f"  Using {weights}-weighted plot")
    print(f"  Snapshots: {snap_offset} to {snap_offset + num_snaps}")
    print(f"  Output: {os.path.join(save_path, movie_name) if save_path else 'interactive window'}")
    
    # Load first snapshot to check if default center/box needed
    o = arun.Run(snappath=snap_path, snapbase="snap_")
    s = o.loadSnap(snapnum=snap_offset)
    
    # Set defaults if not provided
    if center is None:
        if hasattr(s, 'header') and hasattr(s.header, 'BoxSize'):
            center = [s.header.BoxSize / 2.0] * 3
        else:
            center = [500, 500, 500]  # Default fallback
    
    if box is None:
        if hasattr(s, 'header') and hasattr(s.header, 'BoxSize'):
            box = [s.header.BoxSize / 2.0] * 2
        else:
            box = [1000, 1000]  # Default fallback
    
    print(f"  Center: {center}")
    print(f"  Box size: {box}")
    
    # Prepare argument list for all snapshots
    args_list = [
        (i, value, axes_plot, center, box, res, proj, proj_fact, vrange, cmap, logplot, snap_path, weighted, weights)
        for i in range(snap_offset, snap_offset + num_snaps)
    ]
    
    # --- run in parallel ---
    print(f"  Processing {num_snaps} snapshots in parallel using {num_proc} processes...")
    with Pool(processes=num_proc) as pool:
        frames = pool.map(process_snapshot, args_list)
    
    print(f"\n  Processed {len(frames)} snapshots successfully")
    
    # Create animation
    if len(frames) > 0:
        from matplotlib import animation
        
        # Set dark background for animation figure
        plt.style.use('dark_background')
        
        # Create figure for animation
        fig = plt.figure(figsize=(8, 7), dpi=100, facecolor='black')
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        
        # Display first frame
        im = ax.imshow(frames[0])
        ax.set_axis_off()
        
        def animate(frame_num):
            im.set_array(frames[frame_num])
            return [im]
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(frames), 
            interval=1000/fps, blit=True, repeat=True
        )
        
        # Save if requested
        if save_path and movie_name:
            output_file = os.path.join(save_path, movie_name)
            os.makedirs(save_path, exist_ok=True)
            
            # Determine file format and appropriate writer
            _, ext = os.path.splitext(movie_name)
            ext = ext.lower()
            
            print(f"  Saving movie to {output_file}...")
            
            try:
                if ext == '.mp4':
                    # Try ffmpeg first for MP4
                    anim.save(output_file, fps=fps, writer='ffmpeg', savefig_kwargs={'facecolor':'black'})
                elif ext == '.gif':
                    # Use pillow for GIF (most portable)
                    anim.save(output_file, fps=fps, writer='pillow', savefig_kwargs={'facecolor':'black'})
                else:
                    # Default: try to auto-detect
                    anim.save(output_file, fps=fps, savefig_kwargs={'facecolor':'black'})
                
                print(f"  ✓ Movie saved with {fps} fps")
            except RuntimeError as e:
                if ext == '.mp4':
                    print(f"  Warning: ffmpeg not available for MP4")
                    print(f"  Falling back to GIF format...")
                    gif_file = output_file.replace('.mp4', '.gif')
                    anim.save(gif_file, fps=fps, writer='pillow', savefig_kwargs={'facecolor':'black'})
                    print(f"  ✓ Movie saved as {gif_file} with {fps} fps")
                else:
                    raise
        else:
            print("  Displaying interactive movie...")
            plt.show()
    else:
        print("  Error: No frames were processed!")


if __name__ == '__main__':
    # Example usage - density-weighted temperature plot with dark background:
    make_movie(
        value='temp',
        snap_path='/cosma8/data/dp317/dc-naza3/gasCloudNfw/output',
        snap_offset=0,
        num_snaps=20,
        num_proc=32,
        save_path='/cosma8/data/dp317/dc-naza3/gasCloudNfw/plotting',
        movie_name='temperature_test.gif',
        fps=3,
        box=[100, 100],
        vrange=(5e5, 1e9),
        cmap='gnuplot',  # inferno/magma/viridis look great on black
        proj=True,
        proj_fact=0.25,
        res=1024,
        weighted=True,
        weights='temp'
    )