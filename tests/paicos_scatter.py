import paicos as pa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scienceplots

plt.style.use(['science'])

# Adjust settings to reduce verbosity
pa.settings.print_info_when_deriving_variables = False

# Add the missing unit for Arepo standard outputs
pa.add_user_unit("voronoi_cells", "CosmicRaySpecificEnergy", "arepo_velocity**2")

fig, ax = plt.subplots(figsize=(5, 4), dpi=150)

snappath_cr = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/old/output_cr600/'
snappath_homo = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/old/output_homo/'
snapnum = 10

print("Loading snapshots with Paicos...")
s_cr = pa.Snapshot(snappath_cr, snapnum)
s = pa.Snapshot(snappath_homo, snapnum)
s0 = pa.Snapshot(snappath_homo, 0)

print("Calculating...")

# Original logic gets rho0 from t=0 snapshot
rho_0 = s0['0_Density'].value.mean()

# Extract arrays
# Paicos stores raw arrays via .value 
rho_cr = s_cr['0_Density'].value / rho_0
# derive temp array in physical units (Kelvin)
temp_cr = s_cr['0_Temperatures'].to('K').value  

rho_nocr = s['0_Density'].value / rho_0
temp_nocr = s['0_Temperatures'].to('K').value

u_cr = s_cr['0_InternalEnergy'].value
cren_cr = s_cr['0_CosmicRaySpecificEnergy'].value
c_ratio = cren_cr / u_cr

# Box centering 
pos_cr = s_cr['0_Coordinates'].value
center_cr = np.array(s_cr.box_size.value) / 2.0
r_cr = np.linalg.norm(pos_cr - center_cr, axis=1)

pos_nocr = s['0_Coordinates'].value
center_nocr = np.array(s.box_size.value) / 2.0
r_nocr = np.linalg.norm(pos_nocr - center_nocr, axis=1)

mask = (r_nocr >= 0.0078125)
mask_cr = (r_cr >= 0.0078125)

print("Plotting...")

sc_cr = ax.scatter(
    rho_cr[mask_cr], temp_cr[mask_cr], 
    s=0.1, c=c_ratio[mask_cr], cmap='magma', alpha=0.1, norm=LogNorm(vmin=1e-4, vmax=1e2)
)
cbar = fig.colorbar(sc_cr, ax=ax)
cbar.set_label(r'$E_{\rm CR}/E_{\rm th}$', fontsize=11)
cbar.solids.set_alpha(1)

ax.scatter(
    rho_nocr[mask], temp_nocr[mask], 
    s=0.1, color="#4CD4D4", label='Without CR', alpha=0.01
)

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\rho / \rho_0$', fontsize=14)
plt.ylabel(r'$T [K]$', fontsize=14)

ax.legend(fontsize=10, loc='lower left')

plt.tight_layout()

outpath = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/snap-plotting/tests/pap/tempvsrho_comp_paicos.png'
plt.savefig(outpath, dpi=300)
print(f"Saved plot to {outpath}")
