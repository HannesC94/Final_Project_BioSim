import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def add_rec_patch(ax, idx_rec, dwidth=0, **kwargs):
    '''
        add a rectangle to an existing axis object

        Parameters
        ----------
        ax: matplotlib.axes
            axes to which the rectangle is added
        idx_rec: list, [[phi_min, phi_max], [psi_min, psi_max]]
            contains the indices, which specify the rectangle
        dwidth: float
            shrink the rectangle on all sides to identify two rectangles
            next to each other better
    '''

    idx_rec_np = np.array(idx_rec)
    ll = (idx_rec_np[0, 0]+dwidth, idx_rec_np[1, 0]+dwidth)
    width = (idx_rec_np[0, 1] - idx_rec_np[0, 0])-2*dwidth
    height = (idx_rec_np[1, 1] - idx_rec_np[1, 0])-2*dwidth
    rect = patches.Rectangle(ll, width, height,
                             facecolor='none', **kwargs)
    ax.add_patch(rect)


def where_in_rec(phi, psi, rec_list):
    '''
        calculates boolean. True, if [phi, psi] are in a a rectangle.

        Parameters
        ----------
        phi: ndarray, shape(N,2)
            array containing phi values in second column and time in first column
        psi: ndarray, shape(N,2)
            array containing psi values in second column and time in first column
        rec_list: list
            list of lists, which contain the indices, that define the rectangles
            [[[phi_min1, phi_max1],[psi_min1, psi_max1]],
             [[phi_min2, phi_max2],[psi_min2, psi_max2]] ]

        Returns
        -------
        idx_state: ndarray, shape(N,)
            boolean array. True where [phi[i,1], psi[i,1]] is in any of the
            rectangles, specified in rec_list
    '''
    phi_in_rec_list = []
    psi_in_rec_list = []
    both_in_rec_list = []
    for rec in rec_list:
        phi_in_rec = np.asarray((phi[:, 1] >= rec[0][0]) & (phi[:, 1] < rec[0][1]))
        psi_in_rec = np.asarray((psi[:, 1] >= rec[1][0]) & (psi[:, 1] < rec[1][1]))
        phi_in_rec_list.append(phi_in_rec)
        psi_in_rec_list.append(psi_in_rec)
        both_in_rec_list.append(np.logical_and(phi_in_rec, psi_in_rec))
    idx_state = np.logical_or.reduce(np.array(both_in_rec_list))
    return idx_state


def ramachandran_plot(phi, psi, xy_idx=[0.5, 0.5], figsize=[12, 5], bins=100):
    '''
        Plot a histogram of Phi and Psi along with a horizontal and a vertical
        line in the histogram. The occupation propability along these lines is
        plotted in a second plot.

        Parameters
        ----------
        phi: ndarray, shape(N,2)
            array with time in first column and phi angles in second column
        psi: ndarray, shape(N,2)
            array with time in first column and psi angles in second column
        xy_idx: list or tuple with values from [0, 1]
            contains the position along x (phi) and y(psi) axis, where the
            vertical and horizontal lines are shown. Value is fraction of
            the respective axis length.
        figsize: tuple/list
            figure size, passed to plt.subplots()
        bins: int
            number of bins in histogram

    '''
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Ramachandran plot and occupation probability\n along two chosen lines.')
    # create histogram data for phi/psi plot
    histogram = ax1.hist2d(phi[:, 1], psi[:, 1], bins=bins, density=True)
    h = histogram[0]
    phi_edges = histogram[1]
    psi_edges = histogram[2]
    phi_idx = int(bins*xy_idx[0])
    psi_idx = int(bins*xy_idx[1])
    phi_c = phi_edges[phi_idx]
    psi_c = psi_edges[psi_idx]

    ax1.hlines(y=psi_c, xmin=-np.pi, xmax=np.pi, color='b')
    ax1.vlines(x=phi_c, ymin=-np.pi, ymax=np.pi, color='r')
    ax1.set_xlabel(r'$\Phi$ [rad]')
    ax1.set_ylabel(r'$\Psi$ [rad]')
    # add_rec_patch(ax1, [[-np.pi, np.pi], [-2, 1.3]])
    ax2.plot(phi_edges[:-1], h[:, psi_idx], label=r'$\Psi$ :{:.3f}'.format(psi_c))
    ax2.plot(psi_edges[:-1], h[phi_idx, :], label=r'$\Phi$ :{:.3f}'.format(phi_c), color='r')
    ax2.set_ylim(0, np.max(h))
    ax2.set_xlabel(r'$\Phi/\Psi$ [rad]')
    ax2.set_ylabel('occupation propability along line')
    fig.colorbar(histogram[3], ax=ax1)

    plt.legend()
    # plt.tight_layout()
    # plt.show()
    return fig, ax1, ax2


def T_of_tau(s, n_tau, K):
    '''
        calculates the transition matrix T(tau) of a state trajectory s(t) with a time lag of tau = n_tau*dt

        Parameters
        ----------
        s: ndarray, shape(N,2)
            state vs time trajectory. Occupied state in second column
        n_tau: int
            integer between 1 and N. corresponds to a time lag of tau=n_tau*dt, where dt is the time step in the state trajectory
        K: integer
            dimension of (quadratic) matrix T(tau). Corresponds to number of different states in s
    '''
    N_ij = np.zeros((K, K))
    for i in np.arange(K)+1:
        delta_si = np.array(s[:, 1] == i).astype(int)
        for j in np.arange(K)+1:
            delta_sj = np.array(s[:, 1] == j).astype(int)
            N_ij[i-1, j-1] = delta_si[:-n_tau].dot(delta_sj[n_tau:])
    # normalize N_ij
    T_ij = N_ij/np.sum(N_ij, axis=1)[:, np.newaxis]
    return T_ij


def get_lambda2(A):
    """
    Calculates the left eigenectors and eigenvalues of A. Returns the second largest eigenvalue.
    """
    eig_values, eig_vectors = np.linalg.eig(A.T)
    # numpy sort returns sorted values in increasing order
    lambda2 = np.sort(eig_values)[-2]
    return lambda2


def calc_timp(tau_range, s, K):
    '''
        calculates the evolution of the implicit time scale with respect to differentlag times.
        The
        Parameters
        ----------
        tau_range: list/array_like
            list of integers n_tau, corresponding to lag time n_tau*dt
        s: ndarray, shape(N,2)
            state trajectory
        K: int
            number of different states in s

        Returns
        -------
        timp: ndarray, shape(tau_range.shape())
            array with calculated implicit time scales. To get the time, one needs
            to multiply by the time step from the state trajectory.
    '''
    timp = np.empty(len(tau_range))
    for i, tau in enumerate(tau_range):
        T_tau = T_of_tau(s, tau, K)
        lambda2 = get_lambda2(T_tau)
        timp[i] = - tau/np.log(lambda2)
    return timp


def add_timp_plot(ax, tau_range, s, K=3,  **kwargs):
    '''
        add a plot of the implicit time scale to an axes

        Parameters
        ----------
        tau_range: list/array_like
            list of integers n_tau, corresponding to lag time n_tau*dt
        s: ndarray, shape(N,2)
            state trajectory
        K: int
            number of different states in s
    '''
    dt = np.mean(np.diff(s[:, 0]))
    t_imp = calc_timp(tau_range, s, K)
    ax.plot(tau_range*dt, t_imp*dt, **kwargs)
    ax.set_xlabel(r'$\tau$ [ps]')
    ax.set_ylabel(r'$t_{impl}$ [ps]')


def plot_timp(tau_range_1, tau_range_2, s, axh_y=None, axv_x=None,  **kwargs):
    '''
        Plot implicit time scale, given a state trajectory s for two ranges of lag times.

        Parameters
        ----------
        tau_range1/2: list/array_like
            list of integers n_tau, corresponding to lag time n_tau*dt
        s: ndarray, shape(N,2)
            state trajectory

    '''
    dt = np.mean(np.diff(s[:, 0]))
    fig, [ax1, ax2] = plt.subplots(1, 2, **kwargs)
    add_timp_plot(ax1, tau_range_2, s, ls='-')
    ax1.axvspan(xmin=tau_range_1[0]*dt, xmax=tau_range_1[-1]
                * dt, color='y', label='region of interest')
    ax1.legend(fontsize=15)
    add_timp_plot(ax2, tau_range_1, s, ls=':', marker='x')
    ax2.axvline(x=axv_x)
    ax2.axhline(y=axh_y)
    fig.suptitle(r'Implie   d time scale as a function of the lag time $\tau$', fontsize=15)
    plt.show()


def compare_timp(tau_range, state_trajs, **kwargs):
    '''
        Plot implicit time scales as a function of the lag time for different state trajectories.

        Parameters
        ----------
        tau_range: list/array_like
            list of integers n_tau, corresponding to lag time n_tau*dt
        state_trajs: ndarray, shape(N,2)
            state trajectories

    '''

    fig, ax = plt.subplots()
    for i, s in enumerate(state_trajs):
        add_timp_plot(ax, tau_range, s, label='State def {}'.format(i+1), **kwargs)
    ax.legend()
    fig.suptitle('Comparison of implied time scales\nfor different core definitions')
    plt.show()


def shrink_rec(rec, ds):
    '''
        shrink a rectangle by ds on every side.
        Parameters
        ----------
        rec: list
            if rec is a list of lists, containing the indices, every rectangle is
            made smaller by the same amount.
        ds: float
            ds is subtracted from the upper boundaries and substracted from the
            lower boundaries of the rectangle(s)

        Return
        ------
        rec: list, same shape as input rec
            contains the new indices for the smaller rectangles
    '''
    rec = np.array(rec)
    if len(rec.shape) == 2:
        rec[:, 0] += ds
        rec[:, 1] -= ds
    elif len(rec.shape) == 3:
        rec[:, :, 0] += ds
        rec[:, :, 1] -= ds
    return rec.tolist()


def make_state_traj(phi, psi, state_dict, core_def=True):
    '''
        Create a state trajectory s(t) based on Phi and Psi angles. For every state there is a list with indices, specified in state_dict. The indices specify the corner of the rectangles which have been assigned to the state. s(t) will have the state value, that corresponds to the rectangle in which (phi(t), psi(t)) lies.

        Parameters
        ----------
        phi/psi: ndarray, shape(N,2)
            angles over time
        state_dict: dictionary
            keys: different states
            values: nested lists, containing indices, that specify rectangles, which are assigned to the state
        core_def: bool
            if True: checks the state trajectory for  entries without state assignement and assigns them to the previously visited state

        Return
        ------
        s: ndarray, shape(N,2)
            state trajectory, contains time in first dimension and one of the states in the second dimension
    '''
    # create state trajectory (states vs. time)
    s = np.zeros(psi.shape)
    # set first axis to time
    s[:, 0] = psi[:, 0]
    for i, state in enumerate(state_dict.keys()):
        rec_list = state_dict[state]
        idx_state = where_in_rec(phi, psi, rec_list)
        s[idx_state, 1] = i+1
    if core_def:
        for i in range(len(s)):
            if s[i, 1] == 0:
                s[i, 1] = s[i-1, 1]
            else:
                continue
    return s


def visualize_states(rect_dict, phi, psi, color_list, ax=None):
    '''
    Creates a Ramachandran plot and adds rectangles to it with an indicator which state the rectangles belong to.

    Parameters
    ----------
    rect_dict: dictionary
    contains the indices, defining the rectangles, the states are assigned to
    phi/psi: ndarray, shape(N,2)
    angles for the ramachandran plot
    color_list: list
    list of edgecolors for the rectagles, same number of entries as there are differen states
    '''
    colors = dict()
    for i, key in enumerate(rect_dict.keys()):
        colors[key] = color_list[i]
    if ax == None:
        fig, ax = plt.subplots()
    ax.set_xlabel(r'$\Phi$ [rad]', size=12)
    ax.set_ylabel(r'$\Psi$ [rad]', size=12)
    ax.set_title(r'Definition of State regions')
    hist = ax.hist2d(phi[:, 1], psi[:, 1], bins=100, density=True)
    for state in rect_dict.keys():
        state_recs = rect_dict[state]
        c = colors[state]
        for rec in state_recs:
            add_rec_patch(ax, rec, dwidth=0.02, edgecolor=c, lw=3)
    ax.text(2, 2, 'st1', color=colors['state1'], size=18)
    ax.text(-0.5, 2, 'st2', color=colors['state2'], size=18)
    ax.text(1, -.5, 'st3', color=colors['state3'], size=18)
    if ax == None:
        plt.show()


def calc_MSM_validation_data(n_tau_list, s, K):
    '''
        Takes a list of lag times (in form of integer multiples of dt) and
        determines all mulitples, m, of that lag times, which are still in the time
        range of the state trajectory s. Then a set of transition matrices, of these
        lag times is calculated and risen to the power of all m's, that are specific
        to that lag time.

        Parameters
        ----------

        Return
        ------
        ref_data: list
            ref_data[0]: integer values n_t which corresponds to lag times tau_t
            ref_data[1]: transition matrices T(tau_t)=T(n_t*dt)
        tau_data: dictionary
            For all entries in n_tau_list there is an entry in tau_data. For every
            lag time tau_data contains a list with two entries.
            tau_data[0]: list of m*n_tau for all different powers m of n_tau
            tau_data[1]: ndarray, transition matrices T(n_tau)**m to the power of m
    '''
    num_tau_powers = len(n_tau_list)
    s_n = np.copy(s)
    # change the time values of s to integer multiples of dt
    s_n[:, 0] = s_n[:, 0]/np.mean(np.diff(s[:, 0]))
    T_tau_list = [T_of_tau(s, tau, K) for tau in n_tau_list]

    tau_data = {str(tau): [] for tau in n_tau_list}
    n_t_idx = []
    for i, n_tau in enumerate(n_tau_list):
        # generate boolean array at which time steps (n_t) n_t is a multiple of
        # the lag time n_tau
        tau_fold_idx = (np.mod(s_n[:, 0], n_tau) == 0)
        n_t_idx.append(tau_fold_idx)
        n_tau_powers = (s_n[tau_fold_idx, 0]/n_tau).astype(int)
        # remove first entry, because it corresponds to n_t=0
        n_tau_powers = n_tau_powers[n_tau_powers != 0]
        T_to_m = np.zeros((len(n_tau_powers), K, K))
        T_n_tau = T_tau_list[i]
        for i, m in enumerate(n_tau_powers):
            T_to_m[i] = np.linalg.matrix_power(T_n_tau, m)
        # NOTE: multiply n_tau_powers with ntau to get the right times for the plot in the end
        tau_data[str(n_tau)] = [n_tau_powers*n_tau, np.copy(T_to_m)]

    idx_n_t = np.logical_or.reduce((n_t_idx))
    n_t_list = (s_n[idx_n_t, 0]).astype(int)
    n_t_list = n_t_list[n_t_list != 0]
    T_n_t = np.zeros((len(n_t_list), K, K))
    for i, n_t in enumerate(n_t_list):
        T_n_t[i] = T_of_tau(s, n_t, K)
    ref_data = [n_t_list, T_n_t]

    return [ref_data, tau_data]


def add_MSM_val(tau_data, ref_data, ax, transition, dt=0.2, ref_points=30, **kwargs):
    '''
        Plot one of  elements of the transition matrices, calculated with calc_MSM_validation_data on an axis.

        Parameters
        ----------
        ref_data: list
            ref_data[0]: integer values n_t which corresponds to lag times tau_t
            ref_data[1]: transition matrices T(tau_t)=T(n_t*dt)
        tau_data: dictionary
            tau_data[0]: list of m*n_tau for all different powers m of n_tau
            tau_data[1]: ndarray, transition matrices T(n_tau)**m to the power of m
        ax: axis object
            ax t plot on
        transition: tuple(i,j) or int
            which entry (p_ij) from the transition matrices shall be taken an plotted against time
            if int, the diagonal elemnt is taken
        dt: float
            time between frames in ps
        NOTE: For more info, see calc_MSM_validation_data docstring.
    '''

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # make numpy array out of tuple (transition)
    if type(transition) == int:
        trans_idx = np.array((transition, transition)) - 1
    else:
        trans_idx = np.array(transition) - 1

    # pick elements from ref_data[0], so that you get linear spacing in logarithmic representation
    n_t = ref_data[0]
    idx_max = np.log10(len(n_t)-1)
    n_t_pick_log = (10**np.linspace(0, idx_max, ref_points)).astype(int)
    ax.scatter(n_t[n_t_pick_log]*dt, ref_data[1][n_t_pick_log, trans_idx[0], trans_idx[1]],
               label='MD data', s=50, marker='s', facecolors='none', edgecolors='k', alpha=0.3)
    for n_tau in tau_data.keys():
        n_m_dt, T_to_m = tau_data[n_tau]
        # place a text box in upper left in axes coords
        ax.text(0.9, 0.9, 'State '+str(transition), transform=ax.transAxes, fontsize=14,
                verticalalignment='top', ha='right', bbox=props)
        # plot data for n_tau
        ax.plot(n_m_dt*dt, T_to_m[:, trans_idx[0], trans_idx[1]],
                label=r'$\tau$ = {:.0f} ps'.format(int(n_tau)*dt))


def plot_MSM_val(data_matrix, to_state='same', figsize=(11.69, 12), ref_points=30):
    title_string = 'Validation of MSM model. Every column shows the results of a different state defition. '

    grid_dict = {
        'wspace': 0.0,
        'hspace': 0.0
    }
    # Parameters passed on to the subplots call
    params = {'xscale': 'log',
              'xlim': (1, 400),
              }

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=figsize,
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
            if to_state == 'same':
                trans = state
            else:
                trans = np.array((state, to_state))
            add_MSM_val(tau_data, ref_data, ax, transition=trans, ref_points=ref_points)
    # create legend above all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, title=title_string, fontsize='x-large')
    plt.show()
