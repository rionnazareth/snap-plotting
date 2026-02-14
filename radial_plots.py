import arepo_run as arun
import matplotlib.pylab as plt
import numpy as np
import gadget
import yt
import cmasher
import h5py

#Some constants
gamma    = 5./3
unit_m   = 1.989e43
unit_v   = 1.e5
unit_l   = 3.09567758e21
unit_t   = unit_l/unit_v
unit_rho = unit_m/unit_l**3

def radial_profile(s, value, radial_range=None, nbins=50, post_shock=False, shock_path=None):
    """
    Bin 3D data into radial bins assuming spherical symmetry.

    Parameters:
    -----------
    s : object
        Snapshot object containing data
    positions : array
        Particle positions (N, 3)
    value : str
        Key of quantity to bin (e.g., 'density', 'temperature')
    radial_range : tuple, optional
        (r_min, r_max) in code units. If None, uses full range
    nbins : int
        Number of radial bins

    Returns:
    --------
    r_bin : array
        Radial bin centers
    value_bin : array
        Binned quantity (mean values)
    """
    positions = s.pos
    #calcualting temperature
    if value == 'temp':
        kB       = 1.381e-16
        mP       = 1.66e-24
        xH       = 0.76
        meanMolecularWeight = 0.6*mP #4* mP / (1 + 3*xH + 4*xH * s.data['ne'])#/(s.data['rho'] * unit_rho * xH/mP))
        s.data['temp']      = (gamma - 1) * meanMolecularWeight / kB * s.data['u'] * unit_v**2

    #calculating speed
    if value == 'speed':
        s.data['speed']     = np.linalg.norm(s.data['vel'], axis=1)

    #calculationg vorticity magnitude
    if value == 'vortmag':
        s.data['vortmag']     = np.linalg.norm(s.data['vortmag'], axis=1)
    
    #calculating density gradient to find shocks (grad rho/rho<<1 for sound waves)
    if value == 'grar_rho':
        s.data['grar_rho']=np.linalg.norm(s.data['grar'],axis=1)/s.data['rho']

    if value == 'energdens':
        s.data['energdens'] = s.data['u']*s.data['rho']
        
    if value == 'bflds':
        s.data['bflds'] = np.linalg.norm(s.data['bfld'], axis=1)

    if post_shock:
        with h5py.File(shock_path, "r") as shocks_file:
            s.data['shocks_coords']     = shocks_file["Coordinates"][:]
            s.data['temperature']       = shocks_file["Temperature"][:]
            s.data['preshock_temp']     = shocks_file["PreShockTemperature"][:]
            s.data['mach']              = shocks_file["Machnumber"][:]
            s.data['shock_direction']   = shocks_file["ShockDirection"][:]
            s.data['preshock_rho']      = shocks_file["PreShockDensity"][:] * unit_rho
            s.data['postshock_rho']     = shocks_file["PostShockDensity"][:] * unit_rho
            s.data['preshock_p']        = shocks_file["PreShockPressure"][:]
            s.data['postshock_p']       = shocks_file["PostShockPressure"][:]
            s.data['preshock_v']        = shocks_file["PreShockVelocity"][:]
            s.data['postshock_v']       = shocks_file["PostShockVelocity"][:]
            s.data['surf']              = shocks_file["Surface"][:]    
            s.data['uflux']         = shocks_file["GeneratedInternalEnergyFlux"][:]
            s.data['edis']          = s.data['uflux']*s.data['surf'] 
        
        # Find the mapping from shocks_coords to pos
        from scipy.spatial import cKDTree

        # Build a KDTree for fast nearest neighbor lookup
        tree = cKDTree(s.data['shocks_coords'])
        _, indices = tree.query(s.data['pos'])

        # Reorder all shock parameters to match pos ordering
        shock_params = ['temperature', 'preshock_temp', 'mach', 'shock_direction', 
                        'preshock_rho', 'postshock_rho', 'preshock_p', 'postshock_p',
                        'preshock_v', 'postshock_v', 'surf', 'uflux', 'edis']

        for param in shock_params:
            if param in s.data:
                s.data[param] = s.data[param][indices]

        # Now shocks_coords should match pos
        s.data['shocks_coords'] = s.data['shocks_coords'][indices]
    # Center of the box
    center = np.array([s.boxsize/2, s.boxsize/2, s.boxsize/2])
    # Calculate radial distance from box center
    r = np.linalg.norm(positions - center, axis=1)

    s.data['vrad'] = np.sum((positions - center) * s.data['vel'], axis=1) / r    
    
    # Set radial range
    if radial_range is None:
        r_min, r_max = r.min(), r.max()
    else:
        r_min, r_max = radial_range
    
    # Create radial bins
    r_bins = np.linspace(r_min, r_max, nbins + 1)
    r_bin = (r_bins[:-1] + r_bins[1:]) / 2
    
    # Bin the quantity
    value_bin, _ = np.histogram(r, bins=r_bins, weights=s.data[value])
    counts, _ = np.histogram(r, bins=r_bins)
    value_bin /= counts
    
    return r_bin, value_bin

def plot_normalized_comparison(r_bin, y1_values, y2_values, 
                                y1_label='Energy Density', y2_label='Density',
                                xlabel='Radius [kpc]',
                                ylabel='Normalized Value',
                                title='Radial Profile Comparison',
                                figsize=(10, 6),
                                colors=('#1f77b4', '#ff7f0e'),
                                show_ranges=True,
                                logplot=False, norm=True, newfig=True):
    """
    Create a publication-quality comparison plot of two normalized quantities.
    
    Parameters:
    -----------
    val1 : str
        First quantity to plot
    val2 : str
        Second quantity to plot
    y1_label : str
        Label for first quantity
    y2_label : str
        Label for second quantity
    xlabel : str, optional
        X-axis label (default: 'Radius [kpc]')
    ylabel : str, optional
        Y-axis label (default: 'Normalized Value')
    title : str, optional
        Plot title (default: 'Radial Profile Comparison')
    figsize : tuple, optional
        Figure size (default: (10, 6))
    colors : tuple, optional
        Colors for the two lines (default: ('#1f77b4', '#ff7f0e'))
    show_ranges : bool, optional
        Print original value ranges (default: True)
    norm : bool, optional
        Whether to normalize the values (default: True)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Reset matplotlib to defaults first to clear any previous settings
    plt.rcdefaults()
    
    # Set LaTeX-style rcParams for publication-quality plots
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
        'text.usetex': False,
        'mathtext.fontset': 'cm',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': figsize,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 2.5,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
    })  


    
    # Normalize both quantities (subtract min, divide by range)
    if norm:
        y1_norm = (y1_values - np.nanmin(y1_values)) / (np.nanmax(y1_values) - np.nanmin(y1_values))
        y2_norm = (y2_values - np.nanmin(y2_values)) / (np.nanmax(y2_values) - np.nanmin(y2_values))
    else:
        y1_norm = y1_values
        y2_norm = y2_values
    
    # Create the plot
    if newfig:
        fig, ax = plt.subplots(figsize=figsize)
    if logplot:
        plt.semilogy(r_bin, y1_norm, label=f'{y1_label}', 
                color=colors[0], linewidth=2.5, alpha=0.9)
        plt.semilogy(r_bin, y2_norm, label=f'{y2_label} ', 
                color=colors[1], linewidth=2.5, alpha=0.9)

    else:
        plt.plot(r_bin, y1_norm, label=f'{y1_label}', 
                color=colors[0], linewidth=2.5, alpha=0.9),
        plt.plot(r_bin, y2_norm, label=f'{y2_label}', 
                color=colors[1], linewidth=2.5, alpha=0.9)
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, pad=15)
    plt.legend(loc='best', frameon=True, shadow=True, fancybox=True)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Set the spine width
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    
    # Print normalization ranges for reference
    if show_ranges:
        print(f"{y1_label} - Original range: [{np.nanmin(y1_values):.3e}, {np.nanmax(y1_values):.3e}]")
        print(f"{y2_label} - Original range: [{np.nanmin(y2_values):.3e}, {np.nanmax(y2_values):.3e}]")
    
    return plt.gcf(), plt.gca()

if __name__ == "__main__":
    # Example usage
    o  = arun.Run(snappath='/home/c5046973/agn/gasCloudTest/arepo_t/output_cr', snapbase="snap_")
    num = 3
    s_cr  = o.loadSnap(snapnum=num)
    post_shock = False
    shock_path = f'/home/c5046973/agn/gasCloudTest/arepo_t/output_cr/shocks_{num:03d}.hdf5'
    radial_range = (2.5,50)  # in kpc
    r_bin, temp_bin_cr = radial_profile(s_cr, value='speed', radial_range=radial_range, nbins=1000, post_shock=post_shock, shock_path=shock_path)
    r_bin, mach_bin_cr = radial_profile(s_cr, value='mach', radial_range=radial_range, nbins=1000, post_shock=post_shock, shock_path=shock_path)
    # r_bin, vrad_bin = radial_profile(s1, value='vrad', radial_range=radial_range, nbins=1000, post_shock=post_shock, shock_path=shock_path)
    # r_bin, rho_bin = radial_profile(s1, value='rho', radial_range=radial_range, nbins=1000, post_shock=post_shock, shock_path=shock_path)
    
    # plot_normalized_comparison(
    #                                         r_bin=r_bin,
    #                                         y1_values=temp_bin,
    #                                         y2_values=mach_bin,
    #                                         y1_label='Cosmic Ray Energy Density CR',
    #                                         y2_label='Internal Energy Density CR',
    #                                         xlabel='Radius [kpc]',
    #                                         title='Comparison',
    #                                         logplot=False,
    #                                         norm=False,
    #                                         colors=('#d62728', '#2ca02c')  # Red and green
    #                                     )
    o  = arun.Run(snappath='/home/c5046973/agn/gasCloudTest/arepo_t/output_bola', snapbase="snap_")
    num = 3
    s1  = o.loadSnap(snapnum=num)
    post_shock = True
    shock_path = f'/home/c5046973/agn/gasCloudTest/arepo_t/output_bola/shocks_{num:03d}.hdf5'
    # radial_range = (2.5, 100)  # in kpc
    r_bin, temp_bin = radial_profile(s1, value='speed', radial_range=radial_range, nbins=1000, post_shock=post_shock, shock_path=shock_path)
    r_bin, mach_bin = radial_profile(s1, value='mach', radial_range=radial_range, nbins=1000, post_shock=post_shock, shock_path=shock_path)
    # r_bin, vrad_bin = radial_profile(s1, value='vrad', radial_range=radial_range, nbins=1000, post_shock=post_shock, shock_path=shock_path)
    # r_bin, rho_bin = radial_profile(s1, value='rho', radial_range=radial_range, nbins=1000, post_shock=post_shock, shock_path=shock_path)
    
    plot_normalized_comparison(
                                            r_bin=r_bin,
                                            y1_values=mach_bin_cr,
                                            y2_values=mach_bin,
                                            y1_label='Mach number with CR',
                                            y2_label='Mach number without CR',
                                            xlabel='Radius [kpc]',
                                            ylabel='Mach number',
                                            title='Comparison',
                                            logplot=True,
                                            norm=False,
                                            newfig=False,
                                            colors=("#ddae1f", "#20a8ad")  # Red and green
                                        )
    # plot_normalized_comparison(
    #                                         r_bin=r_bin,
    #                                         y1_values=rho_bin,
    #                                         y2_values=vrad_bin,
    #                                         y1_label='Density',
    #                                         y2_label='Radial Velocity',
    #                                         xlabel='Radius [kpc]',
    #                                         title='Comparison',
    #                                         logplot=False,
    #                                         norm=True,
    #                                         newfig=False,
    #                                         colors=("#9f1bb0", "#17c1db")  # Red and green
    #                                     )

    plt.savefig('plots_new/radial_mach.png', dpi=300)