#!/usr/bin/env python3

'''
Functions to analyze a particle based trajectory.
'''

# TODO Go through and change [x][y][z] notation to [x, y, z].
# TODO Go through and change bead -> particle.

import numpy as np
from math import sqrt
from numba import jit

@jit(nopython=True)
def img_flags_from_traj(traj_wrap, box_config):
    """
    Calculate the image flags of a wrapped particle-based trajectory. This 
    assumes that the dump frequency is sufficiently high such that beads never 
    travel more than half the length of a box dimension, otherwise image flags 
    may be incorrect. Molecules that are spread across a periodic boundary will
    have incorrect image flags, potentially introducing errors in other 
    calculations. Assumes that the box dimensions are constant in time.
    
    Args:
        traj_wrap: The wrapped trajectory of each particle stored as a 3D 
                   numpy array with dimensions 'frame (ascending order) by 
                   particle ID (ascending order) by particle position 
                   (x, y, z)'.
        box_config: The simulation box configuration stored as a 1D numpy array
                    of length 6 with the first three elements being box length
                    (lx, ly, lz) and the last three being tilt factors 
                    (xy, xz, yz).
    Returns:
        img_flags: The image flags of the particles across the trajectory
                   stored as a 3D numpy array with dimensions 'frame (ascending
                   order) by particle ID (ascending order) by particle image 
                   flag (ix, iy, iz)'.
    """

    # TODO Expand to triclinic boxes.
    # TODO Expand to non-constant box dimensions, though not sure how image
    # flags would be packaged in that case.

    # Get simulation parameters from the arguments and preallocate arrays.

    nframes = traj_wrap.shape[0]
    nparticles = traj_wrap.shape[1]
    lx, ly, lz = box_config[0:3]
    img_flags = np.zeros(traj_wrap.shape, dtype=np.int32)

    # Loop through each frame but the first, since changes in position between
    # frames are needed to calculate image flags.

    for frame in range(1, nframes):

        # Loop through each particle's index (index = ID - 1).

        for idx in range(nparticles):

            # Get the particle's change in position from the last frame

            del_x = traj_wrap[frame, idx, 0] - traj_wrap[frame - 1, idx, 0]
            del_y = traj_wrap[frame, idx, 1] - traj_wrap[frame - 1, idx, 1]
            del_z = traj_wrap[frame, idx, 2] - traj_wrap[frame - 1, idx, 2]

            # Store any periodic boundary crossings.

            if del_x > lx / 2:
                img_flags[frame:, idx, 0] -= 1
            if del_y > ly / 2:
                img_flags[frame:, idx, 1] -= 1
            if del_z > lz / 2:
                img_flags[frame:, idx, 2] -= 1
            if del_x < lx / -2:
                img_flags[frame:, idx, 0] += 1
            if del_y < ly / -2:
                img_flags[frame:, idx, 1] += 1
            if del_z < lz / -2:
                img_flags[frame:, idx, 2] += 1

    # Return the image flags.

    return img_flags


@jit(nopython=True)
def unwrap_traj(traj_wrap, box_config, img_flags):
    """
    Calculate the unwrapped trajectory of a particle-based trajectory using the
    trajectory's image flags and the simluation box configuration. Assumes that
    the box dimensions are constant in time.
    
    Args:
        traj_wrap: The wrapped trajectory of each particle stored as a 3D numpy
                   array with dimensions 'frame (ascending order) by particle 
                   ID (ascending order) by particle position (x, y, z)'.
        box_config: The simulation box configuration stored as a 1D numpy array
                    of length 6 with the first three elements being box length
                    (lx, ly, lz) and the last three being tilt factors 
                    (xy, xz, yz)'.
        img_flags: The image flags of the particles across the trajectory
                   stored as a 3D numpy array with dimensions 'frame (ascending
                   order) by particle ID (ascending order) by particle image 
                   flag (ix, iy, iz)'.
    Returns:
        traj_unwrap: The unwrapped trajectory of each particle stored as a 3D 
                     numpy array with dimensions 'frame (ascending order) by 
                     particle ID (ascending order) by particle position 
                     (x, y, z)'.
    """

    # TODO Expand to triclinic boxes.
    # TODO Expand to non-constant box dimensions, though not sure how image
    # flags would be packaged in that case.

    # Get simulation parameters from the arguments and preallocate arrays.

    nframes = traj_wrap.shape[0]
    nparticles = traj_wrap.shape[1]
    lx, ly, lz = box_config[0:3]
    traj_unwrap = np.zeros(traj_wrap.shape)

    # Loop through each frame. 

    for frame in range(nframes):

        # Loop through each particle's index (index = ID - 1).

        for idx in range(nparticles):

            # Unwrap the particle's position based on the image flags and box
            # configuration.

            traj_unwrap[frame, idx, 0] = traj_wrap[frame, idx, 0] +\
                                         img_flags[frame, idx, 0] * lx
            traj_unwrap[frame, idx, 1] = traj_wrap[frame, idx, 1] +\
                                         img_flags[frame, idx, 1] * ly
            traj_unwrap[frame, idx, 2] = traj_wrap[frame, idx, 2] +\
                                         img_flags[frame, idx, 2] * lz

    # Return the unwrapped trajectory.

    return traj_unwrap


@jit(nopython=True)
def mol_com_from_frame(pos, molid, mass):
    """
    Calculate the center of mass of each molecule for a single frame of their
    trajectory.
    
    Args:
        pos: The position of each particle stored as a 2D numpy array with
             dimensions 'particle ID (ascending order) by particle position 
             (x, y, z)'.
        molid: The molecule ID of each particle stored as a 1D numpy array with
               dimensions 'particle ID (ascending order)'.
        mass: The mass of each particle stored as a 1D numpy array with
              dimensions 'particle ID (ascending order)'.

    Returns:
        mol_com: The center of mass of each molecule stored as a 2D numpy array
                 with dimensions 'molecule ID (ascending order) by molecule 
                 center of mass (x, y, z)'.
    """

    # Get simulation parameters from the arguments and preallocate arrays.

    mols = np.unique(molid)
    mol_com = np.zeros((mols.shape[0], 3))

    # Loop through each molecule, find the corresponding particle indices, and
    # then calculate the center of mass of the molecule.

    # TODO There is likely a neater and faster way to do this that does not
    # involve taking into account x, y, and z values separately.

    for mol in mols:
        indices = np.where(molid == mol)
        mol_idx = np.where(mols == mol)[0][0]
        mol_mass = np.sum(mass[indices])
        mol_com_x = np.sum(pos[indices][:, 0] * mass[indices]) / mol_mass
        mol_com_y = np.sum(pos[indices][:, 1] * mass[indices]) / mol_mass
        mol_com_z = np.sum(pos[indices][:, 2] * mass[indices]) / mol_mass
        mol_com[mol_idx, 0] = mol_com_x
        mol_com[mol_idx, 1] = mol_com_y
        mol_com[mol_idx, 2] = mol_com_z

    # Return the molecules' center of masses.

    return mol_com


@jit(nopython=True)
def mol_com_from_traj(traj, molid, mass):
    """
    Calculate the center of mass of each molecule across their trajectory. The
    trajectory can be either wrapped or unwrapped, but in most cases a wrapped
    trajectory will produce incorrect results. 
    
    Args:
        traj: The trajectory of each particle stored as a 3D numpy array with
              dimensions 'frame (ascending order) by particle ID (ascending 
              order) by particle position (x, y, z)'.
        molid: The molecule ID of each particle stored as a 1D numpy array with
               dimensions 'particle ID (ascending order)'.
        mass: The mass of each particle stored as a 1D numpy array with
              dimensions 'particle ID (ascending order)'.

    Returns:
        traj_mol_com: The center of mass of each molecule across their 
                      trajectory stored as a 2D numpy array with dimensions 
                      'frame (ascending order) by molecule ID (ascending order)
                      by molecule center of mass (x, y, z)'.
    """

    # Get simulation parameters from the arguments and preallocate arrays.
    
    nframes = traj.shape[0]
    nmols = np.unique(molid).shape[0]
    traj_mol_com = np.zeros((nframes, nmols, 3))

    # Loop through each frame and get the center of mass of each molecule.

    for frame in range(nframes):
        traj_mol_com[frame] = mol_com_from_frame(traj[frame], molid, mass)

    # Return the molecules' center of masses over the trajectory.

    return traj_mol_com


@jit(nopython=True)
def rg_from_frame(pos, molid, mass, mol_com):
    """
    Calculate the radius of gyration of each molecule for a single frame of
    their trajectory. The radius of gyration is the average distance between
    the particles of a molecule and the molecule's center of mass.
    
    Args:
        pos: The position of each particle stored as a 2D numpy array with
             dimensions 'particle ID (ascending order) by particle position 
             (x, y, z)'.
        molid: The molecule ID of each particle stored as a 1D numpy array with
               dimensions 'particle ID (ascending order)'.
        mass: The mass of each particle stored as a 1D numpy array with
              dimensions 'particle ID (ascending order)'.
        mol_com: The center of mass of each molecule stored as a 2D numpy array
                 with dimensions 'molecule ID (ascending order) by molecule 
                 center of mass (x, y, z)'.

    Returns:
        rg: The radius of gyration of each molecule stored as a 1D numpy array
            with dimensions 'molecule ID (ascending order)'.
    """

    # Get simulation parameters from the arguments and preallocate arrays.

    mols = np.unique(molid)
    rg = np.zeros(mols.shape[0])

    # Loop through each molecule, find the corresponding particle indices, and
    # then calculate the radius of gyration of the molecule.

    for mol in mols:
        indices = np.where(molid == mol)[0]
        mol_idx = np.where(mols == mol)[0][0]
        dr = (pos[indices] - mol_com[mol_idx]).flatten()
        sq_dist = np.dot(dr, dr)
        N = indices.shape[0]
        sq_rg = sq_dist / N
        rg[mol_idx] = sqrt(sq_rg)
        
    # Return the molecules' radii of gyration.

    return rg


@jit(nopython=True)
def rg_from_traj(traj, molid, mass, traj_mol_com):
    """
    Calculate the radius of gyration of each molecule across their trajectory. 
    The trajectory can be either wrapped or unwrapped, but in most cases a 
    wrapped trajectory will produce incorrect results. The radius of gyration 
    is the average distance between the particles of a molecule and the 
    molecule's center of mass.
    
    Args:
        traj: The trajectory of each particle stored as a 3D numpy array with
              dimensions 'frame (ascending order) by particle ID (ascending 
              order) by particle position (x, y, z)'.
        molid: The molecule ID of each particle stored as a 1D numpy array with
               dimensions 'particle ID (ascending order)'.
        mass: The mass of each particle stored as a 1D numpy array with
              dimensions 'particle ID (ascending order)'.
        traj_mol_com: The center of mass of each molecule across their
                      trajectory stored as a 3D numpy array with dimensions 
                      'frame (ascending order) by molecule ID (ascending order)
                      by molecule center of mass (x, y, z)'.

    Returns:
        traj_rg: The radius of gyration of each molecule across their 
                 trajectory stored as a 2D numpy array with dimensions 'frame 
                 (ascending order) by molecule ID (ascending order)'.
    """

    # Get simulation parameters from the arguments and preallocate arrays.
    
    nframes = traj.shape[0]
    nmols = np.unique(molid).shape[0]
    traj_rg = np.zeros((nframes, nmols))

    # Loop through each frame and get the radius of gyration of each molecule.

    for frame in range(nframes):
        traj_rg[frame] = rg_from_frame(traj[frame], molid, mass, 
                                       traj_mol_com[frame])

    # Return the molecules' radii of gyration over the trajectory.

    return traj_rg


# @jit(nopython=True)
# def profile_density(xyz, bead_sel, beads_per_mol, nbins, nsections, ccut,
#                     centering, box_dims, mdata):
# 
#     # TODO Build docstring.
#     # TODO Implement clustering.
#     # TODO Have it use get_mol_com for slab centering.
# 
#     # Filter the trajectory based on the selected beads.
#     
#     xyz_sel = xyz[:, bead_sel, :]
# 
#     # Find the longest dimension of the box, known as the slab dimension.
# 
#     slab_length = max(box_dims)
#     slab_cross_section = 1
#     for i in range(0, len(box_dims)):
#         if box_dims[i] == slab_length:
#             slab_dim = i
#         else:
#             slab_cross_section *= box_dims[i]
#     slab_begin = slab_length / -2
#     slab_end = slab_length / 2
# 
#     # Create histogram bins and dimension array.
# 
#     bin_width = slab_length / nbins
#     bin_volume = bin_width * slab_cross_section
#     dimension = np.arange(slab_begin, slab_end, bin_width)
# 
#     # Calculate the first frame to start, the frames per section, and prepare
#     # arrays for data storage.
# 
#     nframes = xyz.shape[0]
#     nbeads = xyz.shape[1]
#     frames_per_sec = nframes // nsections
#     frame_start = nframes - nsections * frames_per_sec
#     sec_num = 0
#     beads_per_bin = np.zeros((nsections, nbins))
#     slab_molecules_per_frame = np.zeros(nsections * frames_per_sec)
#     offset = np.zeros(3)
# 
#     # Get the center of mass and mass of each molecule if slab centering is 
#     # used and preallocate memory for arrays.
#     
#     if centering == 'SLAB':
#         nmolec = beads_per_mol.shape[0]
#         mol_com = np.zeros((nframes, nmolec, 3))
#         largest_mol = np.amax(beads_per_mol)
#         mol_data = np.zeros((4, largest_mol))
#         for frame in range(nframes):
#             mol_start = 0
#             for mol_num in range(nmolec):
#                 mol_end = beads_per_mol[mol_num] + mol_start 
#                 for bead_num in range(mol_start, mol_end):
#                     mol_data[0][bead_num - mol_start] = xyz[frame][bead_num][0]
#                     mol_data[1][bead_num - mol_start] = xyz[frame][bead_num][1]
#                     mol_data[2][bead_num - mol_start] = xyz[frame][bead_num][2]
#                     mol_data[3][bead_num - mol_start] = mdata[bead_num]
#                 mass_weighted_x = np.sum(mol_data[0] * mol_data[3])
#                 mass_weighted_y = np.sum(mol_data[1] * mol_data[3])
#                 mass_weighted_z = np.sum(mol_data[2] * mol_data[3])
#                 total_mass = np.sum(mol_data[3])
#                 mol_com[frame][mol_num][0] = mass_weighted_x / total_mass
#                 mol_com[frame][mol_num][1] = mass_weighted_y / total_mass
#                 mol_com[frame][mol_num][2] = mass_weighted_z / total_mass
#                 mol_start = mol_end
#                 mol_data.fill(0)
#         mol_mass = np.zeros(nmolec)
#         beads_in_cluster = np.zeros(nmolec)
#         mol_start = 0
#         for mol_num in range(nmolec):
#             mol_end = beads_per_mol[mol_num] + mol_start
#             mol_mass[mol_num] = np.sum(mdata[mol_start : mol_end]) 
#             mol_start = mol_end
# 
#     # Loop through each frame and bin particles. The center of mass determined
#     # by the method string is used as a baseline in the binning process.
#     
#     for frame in range(frame_start, nframes):
# 
#         if centering == 'SYSTEM':
#             xdata = xyz[frame][:, 0]
#             ydata = xyz[frame][:, 1]
#             zdata = xyz[frame][:, 2]
#             mass_weighted_x = np.sum(xdata * mdata)
#             mass_weighted_y = np.sum(ydata * mdata)
#             mass_weighted_z = np.sum(zdata * mdata)
#             total_mass = np.sum(mdata)
#             offset[0] = mass_weighted_x / total_mass
#             offset[1] = mass_weighted_y / total_mass
#             offset[2] = mass_weighted_z / total_mass
#         
#         # The slab method for calculating the positional offset for the system.
#         # Works by finding the largest cluster in the frame, and setting that
#         # cluster's center of mass to be the positional offset. Clusters are 
#         # determined by looking at the distance between the center of mass of 
#         # molecules.
# 
#         if centering == 'SLAB':
# 
#             # Find clusters based on the center of mass of molecules.
# 
#             mol_x = mol_com[frame][:, 0]
#             mol_y = mol_com[frame][:, 1]
#             mol_z = mol_com[frame][:, 2]
# 
#             # Create a contact of the molecules, which details whether a
#             # molecule is within the cluster cutoff (ccut) of other molecules.
# 
#             contact_map = np.zeros((nmolec, nmolec))
#             for i in range(nmolec):                           
#                 i_mol_pos = np.array([mol_x[i], mol_y[i], mol_z[i]])                      
#                 for j in range(i, nmolec):                                                
#                     if i == j:
#                         contact_map[i][j] += 1
#                         continue
#                     j_mol_pos = np.array([mol_x[j], mol_y[j], mol_z[j]])                  
#                     dr = i_mol_pos - j_mol_pos
#                     distance = sqrt(np.dot(dr, dr))
#                     if distance <= ccut:                        
#                         contact_map[i][j] += 1
#                         contact_map[j][i] += 1
# 
#             # Compress the contact map by combining rows with shared contacts.
#             # This causes each row to represent a unique cluster.
#             
#             for row in range(contact_map.shape[0]):
#                 new_contacts = True
#                 while new_contacts:
#                     new_contacts = False
#                     for col in range(contact_map.shape[1]):
#                         if row == col:  # Skip self-contacts.
#                             continue
#                         if (contact_map[row][col] != 0 and 
#                             np.any(contact_map[col])):  # Non-zero row contact.
#                             new_contacts = True
#                             contact_map[row] += contact_map[col]
#                             contact_map[col].fill(0)
# 
#             # From the compressed contact map, find the largest cluster and
#             # set the positional offset to the cluster's center of mass.
# 
#             beads_in_cluster.fill(0)
#             cluster_sizes = np.count_nonzero(contact_map, axis=1)
#             largest_cluster_idx = np.argmax(cluster_sizes)
#             mass_weighted_x = 0
#             mass_weighted_y = 0
#             mass_weighted_z = 0
#             total_mass = 0
#             for col in range(contact_map.shape[1]):
#                 if contact_map[largest_cluster_idx][col] != 0:
#                     mass_weighted_x += mol_x[col] * mol_mass[col]
#                     mass_weighted_y += mol_y[col] * mol_mass[col]
#                     mass_weighted_z += mol_z[col] * mol_mass[col]
#                     total_mass += mol_mass[col]
#             offset[0] = mass_weighted_x / total_mass
#             offset[1] = mass_weighted_y / total_mass
#             offset[2] = mass_weighted_z / total_mass
# 
#         # For each particle of interest, apply the center of mass offset and 
#         # bin it. Periodic boundary conditions are applied when applicable.
# 
#         for bead_num in range(xyz_sel.shape[1]):
#                 bead_pos = xyz_sel[frame][bead_num][slab_dim] -\
#                            offset[slab_dim]
#                 if bead_pos >= slab_end:
#                     bead_pos -= slab_length
#                 if bead_pos < slab_begin:
#                     bead_pos += slab_length
#                 bin_idx = int(((bead_pos / slab_length + 1 / 2) * nbins))
#                 if bin_idx >= nbins or bin_idx < 0:
#                     print('Error: invalid bin index created.')
#                 beads_per_bin[sec_num][bin_idx] += 1
# 
#         # If a section is reached, increment the section ID.
# 
#         if (frame - frame_start + 1) % frames_per_sec == 0:
#             sec_num += 1
#     
#     # Convert binned beads to concentration values and create density profiles.
# 
#     conc = beads_per_bin / frames_per_sec / bin_volume
#     density_profiles = np.zeros((nsections, nbins, 2))
#     for i in range(nsections):
#         density_profiles[i] = np.column_stack((dimension, conc[i]))
# 
#     return density_profiles
# 
# 
# # @jit(nopython=True)
# # def bead_heatmap(xyz, bead_list, fixed_dim, rcut, nbins, box_dims, img_flags):
# #     """
# #     Create a 2D heatmap showing the relative density of nearby beads. The
# #     resolution of the heatmap cover the rij vector (which connects bead 1 to
# #     bead 2) to a maximum of rcut, having free rotation in every dimension but
# #     the fixed dimension given by fixed_dim.
# #     
# #     Args:
# #         xyz: the coordinates of a trajectory stored as a 3D numpy array, where
# #              the dimensions are (in order) simulation frame, atom ID, and
# #              xyz coordinates.
# #         bead_list: 1D numpy array with the index of every bead of interest.
# #         fixed_dim: Dimension to not allow rij to rotate in. Either x, y, or z.
# #                    In the rij vector, it sets that dimension's component to 0.
# #         rcut: maximum length for the rij vector.
# #         nbins: ID of bins for a dimension of the rij vector.
# #         box_dims: the xyz values of the simulation box stored as a 1D numpy
# #                   array, where index 0 is the x-dimension, index 1 is the
# #                   y-dimension, and index 2 is the z-dimension.
# #         img_flags: the trajectory's image flags stored as a 3D numpy array ... 
# #                    (finish and test documentation later).
# # 
# #     Returns:
# #         fig_heatmap, ax_heatmap: matplotlib figure and axes of the heatmap
# #     """
# #     
# #     # TODO Make an rij matrix that is rij_matrix[ri_value][rj_value].
# #     # TODO Go through each rij in the simulation and bin into a bin matrix (of
# #     #      shape equal to rij_matrix). Find closest point.
# #     # TODO Generate a heatmap from the bin matrix.
