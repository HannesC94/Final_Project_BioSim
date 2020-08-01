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

# call change_core to shrink the rectangles
ds_2 = 5/360*2*np.pi  # 5degrees
rect_dict_2_core = state_dicts.change_core(rect_dict_1, ds=ds_2)

# call change_core to shrink the rectangles
ds_3 = 45/360*2*np.pi  # 30degrees
rect_dict_3_core = state_dicts.change_core(rect_dict_1, ds=ds_3)

# create state trajectories
s_1 = make_state_traj(phi, psi, rect_dict_1, core_def=False)
s_2 = make_state_traj(phi, psi, rect_dict_2_core, core_def=True)
s_3 = make_state_traj(phi, psi, rect_dict_3_core, core_def=True)

# %% plot s_1
################################################################################
tau_range_1 = np.arange(1, 200, 5)
tau_range_2 = np.arange(1, 900, 10)
plot_timp(tau_range_1, tau_range_2, s_1, axh_y=49, axv_x=19)

# %% show new state def rectangles
################################################################################
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))

visualize_states(rect_dict_2_core, phi, psi, ax=ax1, color_list=['r', 'g', 'y'])
visualize_states(rect_dict_3_core, phi, psi, ax=ax2, color_list=['r', 'g', 'y'])

plt.tight_layout()
plt.show()


# %% compare timp after core definition
################################################################################
tau_range = np.arange(1, 200, 5)
compare_timp(tau_range, [s_1, s_2, s_3], ls=':', marker='o')


# %% testcell MSM evaluation
################################################################################
tau_list = [2, 5, 20]
dt = 0.2
n_tau_list = [int(tau/dt) for tau in tau_list]

data_1 = calc_MSM_validation_data(n_tau_list, s_1, 3)

data_2 = calc_MSM_validation_data(n_tau_list, s_2, 3)

data_3 = calc_MSM_validation_data(n_tau_list, s_3, 3)

data_matrix = np.array([data_1, data_2, data_3])

# %% plot the MSM validation of all state definitions for all three states
################################################################################
title_string = 'Validation of MSM model. Every column shows the results of a different state defition. '

grid_dict = {
    'wspace': 0.0,
    'hspace': 0.0
}
# Parameters passed on to the subplots call
params = {'xscale': 'log',
          'xlim': (5, 400),
          }

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(11.69, 12),
                         sharey='row', sharex='col', gridspec_kw=grid_dict, subplot_kw=params)
# add empty subplots to generate one x/y-label for all subplots
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel("time in ps", fontsize=16)
plt.ylabel("occupation probability", fontsize=16)
# , bbox=dict(boxstyle='square', facecolor='none', edgecolor='k'))
axes[0, 0].set_title('State definition 1')
# , bbox=dict(boxstyle='square', facecolor='none', edgecolor='k'))
axes[0, 1].set_title('State definition 2')
# , bbox=dict(boxstyle='square', facecolor='none', edgecolor='k'))
axes[0, 2].set_title('State definition 3')
# iterate over the different state definitions (columns)
for def_idx, data in enumerate(data_matrix):
    ref_data = data[0]
    tau_data = data[1]
    # iterate over the different states (rows)
    for state_idx, ax in enumerate(axes[:, def_idx]):
        state = state_idx+1
        add_MSM_val(tau_data, ref_data, ax, transition=(1, state))
# create legend above all subplots
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, title=title_string, fontsize='x-large')
plt.show()


# %%
################################################################################
plot_MSM_val(data_matrix)
#fig, axes = plt.subplots(3,3)
# for i in range(3):
#    for j in range(3):
#        add_MSM_val(data_1[1], data_1[0], axes[i,j], transition=(i,j))
