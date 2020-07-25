import state_dicts
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from evaluation_functions import *
################################################################################
# %% read in data
path_phi = 'simulation_data/ad_angle_PHI.xvg'
path_psi = 'simulation_data/ad_angle_PSI.xvg'
# read in the files for psi and phi
psi_raw = pd.read_csv(path_psi, header=None, skiprows=range(17), sep='\s+')
phi_raw = pd.read_csv(path_phi, header=None, skiprows=range(17), sep='\s+')

################################################################################
# %% change representation of angles to radians and get values of df
psi = psi_raw.values
psi[:, 1] *= 2*np.pi/360
phi = phi_raw.values
phi[:, 1] *= 2*np.pi/360

################################################################################
# %% 2.2 visualisation of psi/phi plane
plot = ramachandran_plot(phi, psi, xy_idx=[0.1, 0.9])

################################################################################
# %% make three state definitions based on assignement to rectangles. these are
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


# ################################################################################
# ## %% 3:
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


################################################################################
# %% plot s_2
tau_range = np.arange(1, 200, 5)
timp_1 = calc_timp(tau_range, s_1, 3)
timp_2 = calc_timp(tau_range, s_2, 3)
timp_3 = calc_timp(tau_range, s_3, 3)

fig, ax1 = plt.subplots()
ax1.plot(tau_range, timp_1, ls=':', marker='x')
ax1.plot(tau_range, timp_2, ls=':', marker='x', color='r')
ax1.plot(tau_range, timp_3, ls=':', marker='x', color='g')


################################################################################
# %% plot rectangles
fig, ax = plt.subplots()
for state in rect_dict_2.keys():
    for rec in rect_dict_2[state]:
        add_rec_patch(ax, rec, dwidth=0.05, edgecolor='r')
ax.plot(np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100))
plt.show()
