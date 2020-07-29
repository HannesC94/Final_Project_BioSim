import state_dicts
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from evaluation_functions import *

# %% read in data and prepare for further analysis
################################################################################
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
plot = ramachandran_plot(phi, psi, xy_idx=[0.3, 0.92])

# %% make three state definitions based on assignement to rectangles.
################################################################################
# the first dictionary is defined in state_dicts in "state_dicts.py"
rect_dict_1 = state_dicts.rect_dict_1
visualize_states(rect_dict_1, phi, psi, color_list=['r', 'g', 'y'])

# call change_core to shrink the rectangles
ds_2 = 5/360*2*np.pi  # 5degrees
rect_dict_2_core = state_dicts.change_core(rect_dict_1, ds=ds_2)
visualize_states(rect_dict_2_core, phi, psi, color_list=['r', 'g', 'y'])

# call change_core to shrink the rectangles
ds_3 = 45/360*2*np.pi  # 30degrees
rect_dict_3_core = state_dicts.change_core(rect_dict_1, ds=ds_3)
visualize_states(rect_dict_3_core, phi, psi, color_list=['r', 'g', 'y'])

# create state trajectories
s_1 = make_state_traj(phi, psi, rect_dict_1, core_def=False)
s_2 = make_state_traj(phi, psi, rect_dict_2_core, core_def=True)
s_3 = make_state_traj(phi, psi, rect_dict_3_core, core_def=True)

# %% plot s_2
################################################################################
tau_range = np.arange(1, 200, 5)
dt = 0.2  # ps
timp_1 = calc_timp(tau_range, s_1, 3)
timp_2 = calc_timp(tau_range, s_2, 3)
timp_3 = calc_timp(tau_range, s_3, 3)

fig, ax1 = plt.subplots()
ax1.plot(tau_range*dt, timp_1*dt, ls=':', marker='x', color='b')
ax1.plot(tau_range*dt, timp_2*dt, ls=':', marker='x', color='r')
ax1.plot(tau_range*dt, timp_3*dt, ls=':', marker='x', color='g')
ax1.set_xlabel(r'$\tau$ [ps]')
ax1.set_ylabel(r'$t_{impl}$ [ps]')

# %% plot rectangles
################################################################################
fig, ax = plt.subplots()
for state in rect_dict_1.keys():
    for rec in rect_dict_1[state]:
        add_rec_patch(ax, rec, dwidth=0.01, edgecolor='g', lw=2)
for state in rect_dict_2_core.keys():
    for rec in rect_dict_2_core[state]:
        add_rec_patch(ax, rec, dwidth=0.05, edgecolor='r')
for state in rect_dict_3_core.keys():
    for rec in rect_dict_3_core[state]:
        add_rec_patch(ax, rec, dwidth=0.05, edgecolor='y')
i = ax.hist2d(phi[:, 1], psi[:, 1], bins=100)
plt.show()

# %% testcell MSM evaluation
################################################################################
n_tau_list = [5, 50, 200]
params = {'xscale': 'log',
          'xlim': (5, 5e3),
          'xlabel': 't [ps]/dt'
          }

data = calc_MSM_validation_data(n_tau_list, s_1, 3)
idx_n_t, ref_data, tau_data = data[0], data[1], data[2]
MSM_validation_plot(tau_data, ref_data, [1, 2, 3], params, figsize=(10, 10))


data_2 = calc_MSM_validation_data(n_tau_list, s_2, 3)
idx_n_t, ref_data, tau_data = data_2[0], data_2[1], data_2[2]
MSM_validation_plot(tau_data, ref_data, [1, 2, 3], params, figsize=(10, 10))


data_3 = calc_MSM_validation_data(n_tau_list, s_3, 3)
idx_n_t, ref_data, tau_data = data_3[0], data_3[1], data_3[2]
MSM_validation_plot(tau_data, ref_data, [1, 2, 3], params, figsize=(10, 10))
