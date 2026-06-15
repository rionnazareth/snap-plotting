import sys
sys.path.insert(0, '/home/dc-naza3/arepo-snap-util/')

import arepo_run as arun
import matplotlib.pylab as plt
import numpy as np
import gadget
import yt
import cmasher
import h5py
from tests.lib import *

#Some constants
gamma    = 5./3
unit_m   = 1.989e43
unit_v =1.651077e6
unit_l   = 3.09567758e21
unit_t   = unit_l/unit_v
unit_rho = unit_m/unit_l**3

# def radial_profile(s, value, radial_range=None, nbins=50, post_shock=False, shock_path=None):
#     """
#     Bin 3D data into radial bins assuming spherical symmetry.

#     Parameters:
#     -----------
#     s : object
#         Snapshot object containing data
#     positions : array
#         Particle positions (N, 3)
#     value : str
#         Key of quantity to bin (e.g., 'density', 'temperature')
#     radial_range : tuple, optional
#         (r_min, r_max) in code units. If None, uses full range
#     nbins : int
#         Number of radial bins

#     Returns:
#     --------
#     r_bin : array
#         Radial bin centers
#     value_bin : array
#         Binned quantity (mean values)
#     """
#     positions = s.pos
#     #calcualting temperature
#     if value == 'temp':
#         kB       = 1.381e-16
#         mP       = 1.66e-24
#         xH       = 0.76
#         meanMolecularWeight = 0.6*mP #4* mP / (1 + 3*xH + 4*xH * s.data['ne'])#/(s.data['rho'] * unit_rho * xH/mP))
#         s.data['temp']      = (gamma - 1) * meanMolecularWeight / kB * s.data['u'] * unit_v**2

#     #calculating speed
#     if value == 'speed':
#         s.data['speed']     = np.linalg.norm(s.data['vel'], axis=1)

#     #calculationg vorticity magnitude
#     if value == 'vortmag':
#         s.data['vortmag']     = np.linalg.norm(s.data['vortmag'], axis=1)
    
#     #calculating density gradient to find shocks (grad rho/rho<<1 for sound waves)
#     if value == 'grar_rho':
#         s.data['grar_rho']=np.linalg.norm(s.data['grar'],axis=1)/s.data['rho']

#     if value == 'energdens':
#         s.data['energdens'] = s.data['u']*s.data['rho']
        
#     if value == 'bflds':
#         s.data['bflds'] = np.linalg.norm(s.data['bfld'], axis=1)

#     if post_shock:
#         with h5py.File(shock_path, "r") as shocks_file:
#             s.data['shocks_coords']     = shocks_file["Coordinates"][:]
#             s.data['temperature']       = shocks_file["Temperature"][:]
#             s.data['preshock_temp']     = shocks_file["PreShockTemperature"][:]
#             s.data['mach']              = shocks_file["Machnumber"][:]
#             s.data['shock_direction']   = shocks_file["ShockDirection"][:]
#             s.data['preshock_rho']      = shocks_file["PreShockDensity"][:] * unit_rho
#             s.data['postshock_rho']     = shocks_file["PostShockDensity"][:] * unit_rho
#             s.data['preshock_p']        = shocks_file["PreShockPressure"][:]
#             s.data['postshock_p']       = shocks_file["PostShockPressure"][:]
#             s.data['preshock_v']        = shocks_file["PreShockVelocity"][:]
#             s.data['postshock_v']       = shocks_file["PostShockVelocity"][:]
#             s.data['surf']              = shocks_file["Surface"][:]    
#             s.data['uflux']         = shocks_file["GeneratedInternalEnergyFlux"][:]
#             s.data['edis']          = s.data['uflux']*s.data['surf'] 
        
#         # Find the mapping from shocks_coords to pos
#         from scipy.spatial import cKDTree

#         # Build a KDTree for fast nearest neighbor lookup
#         tree = cKDTree(s.data['shocks_coords'])
#         _, indices = tree.query(s.data['pos'])

#         # Reorder all shock parameters to match pos ordering
#         shock_params = ['temperature', 'preshock_temp', 'mach', 'shock_direction', 
#                         'preshock_rho', 'postshock_rho', 'preshock_p', 'postshock_p',
#                         'preshock_v', 'postshock_v', 'surf', 'uflux', 'edis']

#         for param in shock_params:
#             if param in s.data:
#                 s.data[param] = s.data[param][indices]

#         # Now shocks_coords should match pos
#         s.data['shocks_coords'] = s.data['shocks_coords'][indices]
#     # Center of the box
#     center = np.array([s.boxsize/2, s.boxsize/2, s.boxsize/2])
#     # Calculate radial distance from box center
#     r = np.linalg.norm(positions - center, axis=1)

#     s.data['vrad'] = np.sum((positions - center) * s.data['vel'], axis=1) / r    
    
#     # Set radial range
#     if radial_range is None:
#         r_min, r_max = r.min(), r.max()
#     else:
#         r_min, r_max = radial_range
    
#     # Create radial bins
#     r_bins = np.linspace(r_min, r_max, nbins + 1)
#     r_bin = (r_bins[:-1] + r_bins[1:]) / 2
    
#     # Bin the quantity
#     value_bin, _ = np.histogram(r, bins=r_bins, weights=s.data[value])
#     counts, _ = np.histogram(r, bins=r_bins)
#     value_bin /= counts
    
#     return r_bin, value_bin

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
    # o  = arun.Run(snappath='/cosma8/data/dp317/dc-naza3/gasCloudNfw/output_crexps', snapbase="snap_")
    # num = 2
    # s_cr  = o.loadSnap(snapnum=num)
    # post_shock = False
    # shock_path = f'/cosma8/data/dp317/dc-naza3/gasCloudNfw/output2/shocks_{num:03d}.hdf5'
    # radial_range = (0,0.01)  # in kpc
    # r_bin, temp_bin_cr = radial_profile(s_cr, value='speed', radial_range=radial_range, nbins=1000, post_shock=post_shock, shock_path=shock_path)
    # r_bin, mach_bin_cr = radial_profile(s_cr, value='mach', radial_range=radial_range, nbins=1000, post_shock=post_shock, shock_path=shock_path)
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
    num = 0
    rad_wind = 0.0078125

    s1  = load_snap_data(num, snappath='/cosma8/data/dp317/dc-naza3/initialConditions/homogeneous/mtests/ics_mt.hdf5', snapbase="snap_", ic=True)
    s2  = load_snap_data(num, snappath='/cosma8/data/dp317/dc-naza3/homogeneous/rhov_hires/5/snap_000.hdf5', snapbase="snap_", ic=True)

    # s1.data['bfld']*=18.55
    # unit_b = np.sqrt(unit_rho * unit_v**2)
    # # B-field specific energy (energy per unit mass in code units -> cgs)
    # b_cgs = s1.data['bfld'] * unit_b
    # rho_cgs = s1.data['rho'] * unit_rho
    # s1.data['bflden'] = np.sum(b_cgs**2, axis=1) / (8 * np.pi * rho_cgs)/(unit_v)**2  # Convert to energy per unit mass in code units    
    # s1.data['bflds'] = np.linalg.norm(s1.data['bfld'], axis=1)*unit_b
    # s1.data['bfldpres'] = s1.data['rho'] * s1.data['bflden']
    # s1.data['xb'] = (s1.data['bfldpres'])/ (s1.data['pres'])
    # s1.data['beta'] = 1/s1.data['xb']

    # s2.data['pres'] = s2.data['rho']*s2.data['u']*(gamma-1)
    # s2.data['bfld']*=18.55
    # unit_b = np.sqrt(unit_rho * unit_v**2)
    # # B-field specific energy (energy per unit mass in code units -> cgs)
    # b_cgs = s2.data['bfld'] * unit_b
    # rho_cgs = s2.data['rho'] * unit_rho
    # s2.data['bflden'] = np.sum(b_cgs**2, axis=1) / (8 * np.pi * rho_cgs)/(unit_v)**2  # Convert to energy per unit mass in code units    
    # s2.data['bflds'] = np.linalg.norm(s2.data['bfld'], axis=1)*unit_b
    # s2.data['bfldpres'] = s2.data['rho'] * s2.data['bflden']
    # s2.data['xb'] = (s2.data['bfldpres'])/ (s2.data['pres'])
    # s2.data['beta'] = 1/s2.data['xb']

    post_shock = False
    shock_path = f'/cosma8/data/dp317/dc-naza3/gasCloudNfw/output2/shocks_{num:03d}.hdf5'
    radial_range = (1e-3, 1.5)  # in kpc
    # unit_l = s1.header['UnitLength_in_cm']
    # s1.data['er'] = s1.data['ne_cm'] / (s1.data['n_dens_cm'])
    v = 'bfld'
    r_bin, temp_bin = radial_profile_lin(s1, field=v, r_range=radial_range, nbins=300, post_shock=post_shock, shock_path=shock_path)
    r_bin2, mach_bin = radial_profile_lin(s2, field=v, r_range=radial_range, nbins=300, post_shock=post_shock, shock_path=shock_path)
    # r_bin, vrad_bin = radial_profile(s1, value='vrad', radial_range=radial_range, nbins=1000, post_shock=post_shock, shock_path=shock_path)
    # r_bin, rho_bin = radial_profile(s1, value='rho', radial_range=radial_range, nbins=1000, post_shock=post_shock, shock_path=shock_path)
    
    plot_normalized_comparison(
                                            r_bin=r_bin,
                                            y1_values=temp_bin,
                                            y2_values=mach_bin,  # Placeholder for Mach number without CR
                                            y1_label='ic',
                                            y2_label='snap 0',
                                            xlabel='Radius [kpc]',
                                            ylabel='Density',
                                            title='Comparison',
                                            logplot=False,
                                            norm=False,
                                            newfig=False,
                                            colors=("#ddae1f", "#20a8ad")  # Red and green
                                        )
    
    plt.axvline(rad_wind, color='k', linestyle='--', linewidth=1.5, label='rad_wind')
    # s1  = load_snap_data(num, snappath='/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/old/output_homo/', snapbase="snap_")
    # post_shock = False
    # shock_path = f'/cosma8/data/dp317/dc-naza3/gasCloudNfw/output2/shocks_{num:03d}.hdf5'
    # radial_range = (1e-10, 2)  # in kpc
    # # r_bin, temp_bin = radial_profile_lin(s1, field='xcr', r_range=radial_range, nbins=300, post_shock=post_shock, shock_path=shock_path)
    # r_bin2, mach_bin = radial_profile_lin(s1, field='wind', r_range=radial_range, nbins=300, post_shock=post_shock, shock_path=shock_path)
    # plot_normalized_comparison(
    #                                         r_bin=r_bin,
    #                                         y1_values=mach_bin,
    #                                         y2_values=mach_bin,
    #                                         y1_label='',
    #                                         y2_label='wind fraction no cr',
    #                                         xlabel='Radius [kpc]',
    #                                         title='Comparison',
    #                                         logplot=False,
    #                                         norm=True,
    #                                         newfig=False,
    #                                         colors=("#9f1bb0", "#17c1db")  # Red and green
    #                                     )

    plt.savefig('/cosma8/data/dp317/dc-naza3/snap-plotting/tests/pap/cr_pass.png', dpi=300)