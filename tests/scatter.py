from lib import *
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

plt.style.use(['science'])

fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

snapbase = 'snap_'
snapnum  = 10

snappath = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/old/output_cr/'
s_cr = load_snap_data(snapnum, snappath=snappath, snapbase=snapbase)

snappath = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/old/output_homo/'
s = load_snap_data(snapnum, snappath=snappath, snapbase=snapbase)

unit_v = s.header['UnitVelocity_in_cm_per_s']
unit_l = s.header['UnitLength_in_cm'] 
unit_m = s.header['UnitMass_in_g']
unit_t = unit_l / unit_v
unit_rho = unit_m / unit_l**3

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

r, left, right = get_shell(s, r_range=(1e-3, 1))
r_cr, left_cr, right_cr = get_shell(s_cr, r_range=(1e-3, 1))

shell_start = r[left]
shell_end = r[right]

shell_cr_start = r_cr[left_cr]
shell_cr_end = r_cr[right_cr]

snappath = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/old/output_cr/'
s0 = load_snap_data(0, snappath=snappath, snapbase=snapbase)
rho_0 = s0.data['rho'].mean()
pres_0 = s0.data['pres'].mean()

r_sh, r_rsh = find_shock_radius(s, r_range=(1e-3, 1), nbins=500)
r_shcr, r_rshcr = find_shock_radius(s_cr, r_range=(1e-3, 1), nbins=500)


# Extract density and temperature (falling back to internal energy 'u' if 'temp' is not present)
rho_cr = s_cr.data['rho'] / rho_0
temp_cr = s_cr.data['temp']
pres_cr = s_cr.data['pres'] / pres_0

rho_nocr = s.data['rho'] / rho_0
temp_nocr = s.data['temp']
pres_nocr = s.data['pres'] / pres_0

rad_wind = 0.0078125

mask = np.ones_like(s.data['r'], dtype=bool)
mask_cr = np.ones_like(s_cr.data['r'], dtype=bool)

mask &=  (s.data['r'] >= rad_wind) 
mask_cr &= (s_cr.data['r'] >= rad_wind) 

# mask &=  (s.data['mach'] == 0) 
# mask_cr &= (s_cr.data['mach'] == 0) 

# mask &= (s.data['r'] >= shell_start) 
# mask_cr &= (s_cr.data['r'] >= shell_cr_start) 

# mask &= (s.data['r'] <= shell_end-0.05) 
# mask_cr &= (s_cr.data['r'] <= shell_cr_end-0.05) 

# mask &= (s.data['r'] >= r_rsh) 
# mask_cr &= (s_cr.data['r'] >= r_rshcr) 
# mask &= (s.data['r'] <= shell_start) 
# mask_cr &= (s_cr.data['r'] <= shell_cr_start) 

# mask &=  (s.data['wind'] > 1e-5)
# mask_cr&= (s_cr.data['wind'] > 1e-5)  

vmin = 1e-2
vmax = 1e2
val = 'ecth'
sc_cr = ax[0].scatter(rho_cr[mask_cr], temp_cr[mask_cr], s=0.1, c=s_cr.data[val][mask_cr], cmap='vanimo', alpha=0.1, norm=LogNorm(vmin=vmin, vmax=vmax))
cbar = fig.colorbar(sc_cr, ax=ax[0])
cbar.set_label(r'$E_{\rm CR}/E_{\rm th}$', fontsize=11)
cbar.solids.set_alpha(1)
# cbar.draw_all()
ax[0].scatter(rho_nocr[mask], temp_nocr[mask], s=0.1, color="#4CD4D4", alpha=0.01)
ax[0].scatter([], [], s=20, color="#4CD4D4", label='Without CR', alpha=1.0)
ax[0].set_xlim((1e-3, 30))
ax[0].set_ylim((1e1, 4e9))

center = [0.5, 0.5, 0.5]#[s_cr.header['BoxSize'] / 2] * 3
box_zoom = [1, 1]
proj_on = False
proj_fact = 0.1
s_cr.axplot_Aweightedslice(ax[1],
                    value=val, weights='rho', cmap='vanimo', colorbar=False, logplot=True, vrange=(vmin, vmax),
                    center=center, box=box_zoom, res=1024,
                    proj=proj_on, proj_fact=proj_fact, axes=[0,2]
                )
ax[1].set_xticklabels(np.round(ax[1].get_xticks()-s_cr.boxsize/2, decimals=2))
ax[1].set_yticklabels(np.round(ax[1].get_yticks()-s_cr.boxsize/2, decimals=2))

# mask_revsh = (s_cr.data['mach'] > 0)&(s_cr.data['r'] <= shell_start)#(s_cr.data['r'] >= 0.99 * r_rshcr) & (s_cr.data['r'] <= 1.01 * r_rshcr)
# mask_sh = (s_cr.data['mach'] > 0)&(s_cr.data['r'] >= shell_start)#(s_cr.data['r'] >= 0.99 * r_shcr) & (s_cr.data['r'] <= 1.01 * r_shcr)

# def plot_hull(ax, x, y, color, label):
#     valid = (x > 0) & (y > 0)
#     x_val = x[valid]
#     y_val = y[valid]
#     if len(x_val) < 3:
#         ax.scatter(x_val, y_val, s=0.1, color=color, label=label, alpha=0.9)
#         return
#     points = np.column_stack((np.log10(x_val), np.log10(y_val)))
#     hull = ConvexHull(points)
#     hull_points = points[hull.vertices]
#     hull_points = np.vstack((hull_points, hull_points[0]))
    
#     # Smooth the polygon
#     distance = np.cumsum(np.sqrt(np.sum(np.diff(hull_points, axis=0)**2, axis=1)))
#     distance = np.insert(distance, 0, 0)
    
#     # Filter points too close to each other to avoid spline error
#     unique_idx = np.concatenate(([True], np.diff(distance) > 1e-6))
#     if unique_idx.sum() > 3:
#         hull_points = hull_points[unique_idx]
#         tck, u = splprep([hull_points[:, 0], hull_points[:, 1]], s=0, per=True)
#         u_new = np.linspace(u.min(), u.max(), 1000)
#         x_new, y_new = splev(u_new, tck)
#         ax.plot(10**x_new, 10**y_new, color=color, label=label, linewidth=0.8)
#     else:
#         ax.plot(10**hull_points[:, 0], 10**hull_points[:, 1], color=color, label=label, linewidth=0.8)

# plot_hull(ax, rho_cr[mask_revsh], temp_cr[mask_revsh], "#4BF848", 'reverse shock')
# plot_hull(ax, rho_cr[mask_sh], temp_cr[mask_sh], "#195900", 'forward shock')

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel(r'$\rho / \rho_0$', fontsize=14)
ax[0].set_ylabel(r'$T [K]$', fontsize=14)
# ax[0].set_title('Temperature vs Density', fontsize=16)

# ax[0].legend(fontsize=10,loc='lower left')

fig.tight_layout()

plt.savefig(f'/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/snap-plotting/tests/meh/crdiff_snapno{snapnum}.png', dpi=300)