import state_dicts
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from evaluation_functions import *

################################################################################
# %% read in data and prepare for further analysis
path_phi = 'simulation_data/ad_angle_PHI.xvg'
path_psi = 'simulation_data/ad_angle_PSI.xvg'
# read in the files for psi and phi
psi_raw = pd.read_csv(path_psi, header=None, skiprows=range(17), sep='\s+')
phi_raw = pd.read_csv(path_phi, header=None, skiprows=range(17), sep='\s+')
# change representation of angles to radians and get values of df
psi = psi_raw.values
psi[:, 1] *= 2*np.pi/360
phi = phi_raw.values
phi[:, 1] *= 2*np.pi/360

# %% 2.2 visualisation of psi/phi plane
################################################################################
plot = ramachandran_plot(phi, psi, xy_idx=[0.1, 0.9])

# %% make three state definitions based on assignement to rectangles. these are
################################################################################
# defines in "state_dicts.py"
# get dictionaries with the rectangles for the 3 different state definitions
# from the python file state_dicts
rect_dict_1 = state_dicts.rect_dict_1
rect_dict_2 = state_dicts.rect_dict_2
rect_dict_3 = state_dicts.rect_dict_3
rect_dict_3['state1']
rect_dict_2['state1']
# create state trajectories
s_1 = make_state_traj(phi, psi, rect_dict_1, core_def=False)
s_2 = make_state_traj(phi, psi, rect_dict_2, core_def=True)
s_3 = make_state_traj(phi, psi, rect_dict_3, core_def=True)


# ## %% 3:
# ################################################################################
# # plot together the occupation probability with the chose rectangles, as well
# # as the phi/psi eig_values
# fig, ax1, ax2 = ramachandran_plot(phi, psi, xy_idx=[0.1, 0.9])
# dw = 0.05
# for rec in rec_list_s1:
#     add_rec_patch(ax1, rec, edgecolor='g', dwidth=dw, linewidth=2)
# for rec in rec_list_s2:
#     add_rec_patch(ax1, rec, edgecolor='c', dwidth=dw, linewidth=2)
# for rec in rec_list_s3:
#     add_rec_patch(ax1, rec, edgecolor='m', dwidth=dw, linewidth=2)
#
# ax2.clear()
# ax2.scatter(phi[idx_state1, 1], psi[idx_state1, 1], label='State 1', color='g')
# ax2.scatter(phi[idx_state2, 1], psi[idx_state2, 1], label='State 2', color='c')
# ax2.scatter(phi[idx_state3, 1], psi[idx_state3, 1], label='State 3', color='m')
# ax2.legend()
# ax2.set_xlabel('Phi [rad]')
# ax2.set_ylabel('Psi [rad]')
#
# plt.show()
#


# %% plot s_2
################################################################################
tau_range = np.arange(1, 200, 5)
dt = 0.2  # ps
timp_1 = calc_timp(tau_range, s_1, 3)
timp_2 = calc_timp(tau_range, s_2, 3)
timp_3 = calc_timp(tau_range, s_3, 3)

fig, ax1 = plt.subplots()
ax1.plot(tau_range*dt, timp_1*dt, ls=':', marker='x')
ax1.plot(tau_range*dt, timp_2*dt, ls=':', marker='x', color='r')
ax1.plot(tau_range*dt, timp_3*dt, ls=':', marker='x', color='g')
ax1.set_xlabel('lag time tau [ps]')
ax1.set_ylabel('implicit time scale [ps]')

# %% plot rectangles
################################################################################
fig, ax = plt.subplots()
for state in rect_dict_2.keys():
    for rec in rect_dict_2[state]:
        add_rec_patch(ax, rec, dwidth=0.05, edgecolor='r')
ax.plot(np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100))
plt.show()

# %% Validation of MSM model
################################################################################
s = np.copy(s_1)
K = 3
n_tau_list = [5, 50, 200]
s_n = np.copy(s_1)
s_n[:, 0] = s_n[:, 0]/np.mean(np.diff(s_1[:, 0]))
T_tau_list = [T_of_tau(s, tau, K) for tau in n_tau_list]

# define index arrays for the different lag times n_tau in n_tau_list
idx_n_1 = np.mod(s_n[:, 0], 5) == 0
n_1_list = (s_n[idx_n_1, 0]/5).astype(int)
n_1_list = n_1_list[1:]
idx_n_2 = np.mod(s_n[:, 0], 50) == 0
n_2_list = (s_n[idx_n_2, 0]/5).astype(int)
n_2_list = n_2_list[1:]
idx_n_3 = np.mod(s_n[:, 0], 200) == 0
n_3_list = (s_n[idx_n_3, 0]/5).astype(int)
n_3_list = n_3_list[1:]
idx_n_t = np.logical_or.reduce([idx_n_1, idx_n_2, idx_n_3])
n_t_list = s_n[idx_n_t, 0].astype(int)
n_t_list = n_t_list[1:]


# list of transition matrices for all relevant times n_t*dt
T_nt_list = np.zeros((len(n_t_list), K, K))
for i, n_t in enumerate(n_t_list):
    T_nt_list[i] = T_of_tau(s, n_tau=n_t, K=K)
n_t_data = [n_t_list, T_n1_list]
n_t_data[0].shape

# generate list of mfolded matrices, defined in T_tau_list
T_n1_list = np.zeros((len(n_1_list), K, K))
for i, n_1 in enumerate(n_1_list):
    T_n1_list[i] = np.linalg.matrix_power(T_tau_list[0], n_1)
n_1_data = [n_1_list, T_n1_list]

T_n2_list = np.zeros((len(n_2_list), K, K))
for i, n_2 in enumerate(n_2_list):
    T_n2_list[i] = np.linalg.matrix_power(T_tau_list[1], n_2)
    n_2_data = [n_2_list, T_n2_list]

T_n3_list = np.zeros((len(n_3_list), K, K))
for i, n_3 in enumerate(n_3_list):
    T_n3_list[i] = np.linalg.matrix_power(T_tau_list[2], n_3)
    n_3_data = [n_3_list, T_n3_list]
# %% plot
################################################################################
fig, ax = plt.subplots()
st = 1
ax.plot(np.log10(n_t_data[0]), n_t_data[1][:, st-1, st-1], marker='o', ls='')
ax.plot(np.log10(n_1_data[0]*5), n_1_data[1][:, st-1, st-1], label='n_tau:{}'.format(n_tau_list[0]))
ax.plot(np.log10(n_2_data[0]*50), n_2_data[1][:, st-1, st-1],
        label='n_tau:{}'.format(n_tau_list[1]))
ax.plot(np.log10(n_3_data[0]*200), n_3_data[1][:, st-1, st-1],
        label='n_tau:{}'.format(n_tau_list[2]))
ax.legend()
ax.set_ylim(0.23, 0.27)
plt.show()
