import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from evaluation_functions import *
# %% read in data
path_phi = 'simulation_data/ad_angle_PHI.xvg'
path_psi = 'simulation_data/ad_angle_PSI.xvg'
# read in the files for psi and phi
psi_raw = pd.read_csv(path_psi, header=None, skiprows=range(17), sep='\s+')
phi_raw = pd.read_csv(path_phi, header=None, skiprows=range(17), sep='\s+')

# %% change representation of angles to radians and get values of df
psi = psi_raw.values
psi[:, 1] *= 2*np.pi/360
phi = phi_raw.values
phi[:, 1] *= 2*np.pi/360


# %% 2.2 visualisation of psi/phi plane
ramachandran_plot(phi, psi, xy_idx=[0.1, 0.9])


# %% 3:
# define indices for state rectangles.
s1_rec_ul = [[-np.pi, -2.07], [1.3, np.pi]]
s1_rec_ur = [[1.3, np.pi], [1.3, np.pi]]
s1_rec_ll = [[-np.pi, -2.07], [-np.pi, -2]]
s1_rec_lr = [[1.3, np.pi], [-np.pi, -2]]
rec_list_s1 = [s1_rec_ul, s1_rec_ur, s1_rec_ll, s1_rec_lr]
idx_state1 = where_in_rec(phi, psi, rec_list_s1)

s2_rec = [[-2.07, 1.3], [-np.pi, -2]]
s2_rec_2 = [[-2.07, 1.3], [1.3, np.pi]]
rec_list_s2 = [s2_rec, s2_rec_2]
idx_state2 = where_in_rec(phi, psi, rec_list_s2)

s3_rec = [[-np.pi, np.pi], [-2, 1.3]]
rec_list_s3 = [s3_rec]
idx_state3 = where_in_rec(phi, psi, rec_list_s3)

# create state trajectory (states vs.)
s = np.zeros(psi.shape)
# set first axis to time
s[:, 0] = psi[:, 0]
# fill up entries of second column with corresponding states at the time
s[idx_state1, 1] = 1
s[idx_state2, 1] = 2
s[idx_state3, 1] = 3

# plot together the occupation probability with the chose rectangles, as well
# as the phi/psi eig_values
fig, ax1, ax2 = ramachandran_plot(phi, psi, xy_idx=[0.1, 0.9])

for rec in rec_list_s1:
    add_rec_patch(ax1, rec, color='g')
for rec in rec_list_s2:
    add_rec_patch(ax1, rec, color='c')
for rec in rec_list_s3:
    add_rec_patch(ax1, rec, color='m')

ax2.clear()
ax2.scatter(phi[idx_state2, 1], psi[idx_state2, 1], label='State 2')
ax2.scatter(phi[idx_state3, 1], psi[idx_state3, 1], label='State 3')
ax2.scatter(phi[idx_state1, 1], psi[idx_state1, 1], label='State 1')
ax2.legend()
ax2.set_xlabel('Phi [rad]')
ax2.set_ylabel('Psi [rad]')

plt.show()

# %% calculate transition matrix


def T_of_tau(s, tau, K):
    N_ij = np.zeros((K, K))
    for i in np.arange(K)+1:
        delta_si = np.array(s[:, 1] == i).astype(int)
        for j in np.arange(K)+1:
            delta_sj = np.array(s[:, 1] == j).astype(int)
            N_ij[i-1, j-1] = delta_si[:-tau].dot(delta_sj[tau:])
    # normalize N_ij
    N_ij = N_ij/np.sum(N_ij, axis=1)[:, np.newaxis]
    return N_ij


def get_lambda2(A):
    """
    Calculates the
    """
    eig_values, eig_vectors = np.linalg.eig(A)
    lambda2 = np.sort(eig_values)[-2]
    return lambda2


def calc_timp(tau_range, s, K):
    timp = np.empty(len(tau_range))
    for i, tau in enumerate(tau_range):
        T_tau = T_of_tau(s, tau, K)
        lambda2 = get_lambda2(T_tau)
        timp[i] = - tau/np.log(lambda2)
    return timp


tau_range = np.arange(1, 300, 2)
timp = calc_timp(tau_range, s, K=3)
timp
plt.plot(tau_range, timp)
psi_raw
