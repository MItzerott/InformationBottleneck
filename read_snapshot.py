'''
To use the `read_gadget` library, either install it via:
`python setup.py install --user`

or make sure that the `src/read_gadget.py` file is in your working
directory.

To find out where the data you are interested in is stored,
check the file using "h5ls filename/Groupname" for data and 
"h5ls -v filename" to list all attributes such as the Header.
Alternatively, consult the Arepo user guide:
https://arepo-code.org/wp-content/userguide/snapshotformat.html
'''

import numpy as np
import read_gadget as rg

# The file path is the full path to the snapshot file but
# excluding the ".0.hdf5" at the end of the file name
def read_snapshot(sim_size, identifier, quiet = False):
	if sim_size == 22:
		filepath = '/home/mitzerott/L50N22/L50N22_10%s/output/snapdir_004/snapshot_004' %identifier
	elif sim_size == 32:
		filepath = '/home/mitzerott/L50N32/L50N32_10%s/output/snapdir_004/snapshot_004' %identifier
	elif sim_size == 64:
		filepath = '/home/mitzerott/L50N32/L50N64_10%s/output/snapdir_004/snapshot_004' %identifier
  
	# To load some data you need to know the name and where it 
	# is stored in the HDF5 file
	pos = rg.read_array(filepath, 'PartType1/Coordinates')
	vel = rg.read_array(filepath, 'PartType1/Velocities')
	ids = rg.read_array(filepath, 'PartType1/ParticleIDs')
	pot = rg.read_array(filepath, 'PartType1/Potential')

	if not quiet:
		print(f'Coordinates shape: {pos.shape} kpc/h')
		print(f'Velocities shape: {vel.shape} km/s')
		print(f'IDs shape: {ids.shape}')

	# For Header information it is similar
	h = rg.read_attribute(filepath, 'HubbleParam')
	z = rg.read_attribute(filepath, 'Redshift')
	box_size = rg.read_attribute(filepath, 'BoxSize') # in kpc/h
	
	if not quiet:
		print(f'Hubble param: {h} km/s/Mpc/100')
		print(f'Redshift: {z}')
		print(f'Box size: {box_size} kpc/h')

	# Some Header information is stored as a list for each particle 
	# type. But we're only interested in PartType1
	npart = rg.read_attribute(filepath, 'NumPart_Total')[1]

	if not quiet:
		print(f'Total particle number: {npart}')
		print(f'Total particle number: {np.round(np.cbrt(npart))}^3')

	# Masses are stored as 1E10 M_sun/h so need to multiply by this factor
	mass_part = rg.read_attribute(filepath, 'MassTable')[1] * 1E10

	if not quiet:
		print(f'Particle mass: {mass_part:e} M_sun/h')

	return pos, vel, pot, ids, h, z, box_size, npart, mass_part
