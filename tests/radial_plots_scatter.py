import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import scienceplots

plt.style.use(['science'])

# Import your library
import lib

def main():
    # 1. Define paths and snapshot number
    # Update these paths to point to your actual simulation data
    snappath = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/old/output_cr' 
    snapbase = 'snap_'
    snapnum = 6  # Replace with the desired snapshot number
    
    # 2. Load the snapshot using lib.py
    # load_snap_data calculates 'temp', 'pres', 'nH_cm', 'vrad', and 'r' automatically
    s = lib.load_snap_data(snapnum, snappath=snappath, snapbase=snapbase)

    unit_v = s.header['UnitVelocity_in_cm_per_s']
    unit_l = s.header['UnitLength_in_cm'] 
    unit_m = s.header['UnitMass_in_g']
    unit_t = unit_l / unit_v
    unit_rho = unit_m / unit_l**3
    
    # # Choose your downsampling factor (e.g., 100 plots 1% of the cells)
    # stride = 100 
    
    # # Extract data for plotting AND apply the stride to each array
    # r = s.data['r'][::stride]
    # nH = s.data['nH_cm'][::stride]
    # vrad = (s.data['vrad'] * unit_v / 1e5)[::stride]  # Convert cm/s to km/s
    # P = s.data['pres'][::stride]
    # T = s.data['temp'][::stride]
    
    # # Choose a variable for coloring the points (apply stride here too!)
    # color_var = s.data['mach'][::stride] if 'mach' in s.data else s.data['speed'][::stride]
    # color_label = '$\\mathcal{M}$' if 'mach' in s.data else 'Speed [km/s]'

    # Define radial range and number of bins for the average
    r_min = np.min(s.data['r'][s.data['r'] > 0])
    r_max = np.max(s.data['r'])
    r_range = (r_min, r_max)
    nbins = 500
    
    # Calculate binned 1D averages using lib.py
    r, nH = lib.radial_profile_log(s, 'nH_cm', r_range=r_range, nbins=nbins)
    _, vrad = lib.radial_profile_log(s, 'vrad', r_range=r_range, nbins=nbins)
    vrad = vrad * unit_v / 1e5  # Convert cm/s to km/s
    _, P = lib.radial_profile_log(s, 'pres', r_range=r_range, nbins=nbins)
    _, T = lib.radial_profile_log(s, 'temp', r_range=r_range, nbins=nbins)
    
    # Average the color variable as well
    color_field = 'xcr'#mach' if 'mach' in s.data else 'speed'
    _, color_var = lib.radial_profile_log(s, color_field, r_range=r_range, nbins=nbins)
    if color_field == 'speed':
        color_var = color_var / 1e5  # Convert to km/s if falling back to speed
    color_label = '$P_\\mathrm{{CR}} / P_\\mathrm{{th}}$'#'$\\mathcal{M}$' if 'mach' in s.data else 'Speed [km/s]'

    # 3. Create the 2x2 plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    
    vmin = 1e-2
    vmax = 1e3
    # Define scatter parameters
    scatter_kwargs = {
        'c': color_var,
        'cmap': 'vanimo', # Adjust cmap to match the image ('nipy_spectral' or 'turbo')
        's': 10,
        'alpha': 0.7,
        'norm': LogNorm(vmin=vmin, vmax=vmax) #if np.nanmin(color_var) > 0 else None
    }
    
    # Top-Left: Density
    sc = axs[0, 0].scatter(r, nH, **scatter_kwargs)
    axs[0, 0].set_ylabel('$n_H \\; [cm^{-3}]$', fontsize=14)
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_ylim(1e-3,2e2)
    
    # Top-Right: Radial Velocity
    axs[0, 1].scatter(r, vrad, **scatter_kwargs)
    axs[0, 1].set_ylabel('$v_\\mathrm{{rad}} \\; [km\\,s^{-1}]$', fontsize=14)
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_ylim(1,2e4)
    
    
    # Bottom-Left: Pressure
    axs[1, 0].scatter(r, P, **scatter_kwargs)
    axs[1, 0].set_ylabel('$P \\; [dyn \\, cm^{-2}]$', fontsize=14)
    axs[1, 0].set_xlabel('$R \\; [kpc]$', fontsize=14)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_ylim(1e-3,2e1)
    
    # Bottom-Right: Temperature
    axs[1, 1].scatter(r, T, **scatter_kwargs)
    axs[1, 1].set_ylabel('$T \\; [K]$', fontsize=14)
    axs[1, 1].set_xlabel('$R \\; [kpc]$', fontsize=14)
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_ylim(1e-3,2e9)
    
    rmin = 0.02
    rmax = 1
    for ax in axs.flat:
        ax.set_xlim(rmin, rmax)
    # 4. Add Colorbars
    # Add one for the top row and one for the bottom row, or just one global
    cbar = fig.colorbar(sc, ax=axs, orientation='vertical', pad=0.02, aspect=40)
    cbar.set_label(color_label, fontsize=14)
    
    # 5. Add reference lines (optional, adjust coefficients as needed)
    # Example: Density scales as r^-2 for a free wind
    r_ref = np.logspace(np.log10(rmin), np.log10(rmax), 50)
    axs[0, 0].plot(r_ref, 1e-4 * r_ref**-2, 'k--', lw=2, label='$\\propto R^{-2}$')
    axs[0, 0].legend(frameon=False, loc='upper right')
    
    # plt.tight_layout()
    plt.savefig(f'/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/snap-plotting/tests/rad/rps_{color_field}_{snapnum}.png', dpi=300, bbox_inches='tight')
    print("Plot saved to radial_profiles_scatter.png")

if __name__ == '__main__':
    main()