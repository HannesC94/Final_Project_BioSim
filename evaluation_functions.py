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
    ax1.set_xlabel('Phi [rad]')
    ax1.set_ylabel('Psi [rad]')
    #add_rec_patch(ax1, [[-np.pi, np.pi], [-2, 1.3]])
    ax2.plot(phi_edges[:-1], h[:, psi_idx], label='Psi:{:.3f}'.format(psi_c))
    ax2.plot(psi_edges[:-1], h[phi_idx, :], label='Phi:{:.3f}'.format(phi_c), color='r')
    ax2.set_ylim(0, np.max(h))
    ax2.set_xlabel('Phi/Psi [rad]')
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
    Calculates the left eigenectors and eigenvalues of A. Returns the second laargest eigenvalue.
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
    '''
    dt = np.mean(np.diff(s[:, 0]))
    t_imp = calc_timp(tau_range, s, K)
    ax.plot(tau_range*dt, t_imp*dt, **kwargs)
    ax.set_xlabel(r'$\tau$ [ps]')
    ax.set_ylabel(r'$t_{impl}$ [ps]')


def plot_timp(tau_range_1, tau_range_2, s, **kwargs):
    dt = np.mean(np.diff(s[:, 0]))
    fig, [ax1, ax2] = plt.subplots(1, 2, **kwargs)
    add_timp_plot(ax1, tau_range_2, s, ls='-')
    ax1.axvspan(xmin=tau_range_1[0]*dt, xmax=tau_range_1[-1]
                * dt, color='y', label='region of interest')
    ax1.legend(fontsize=15)
    add_timp_plot(ax2, tau_range_1, s, ls=':', marker='x')
    ax2.axvline(x=20)
    ax2.axhline(y=48)
    fig.suptitle(r'Implicit time scale as a function of the lag time $\tau$', fontsize=15)
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

    return [idx_n_t, ref_data, tau_data]


def MSM_validation_plot(tau_data, ref_data, states, params=None, **kwargs):
    '''


        Parameters
    '''

    fig, axes = plt.subplots(len(states), 1, subplot_kw=params, **kwargs)
    for id, ax in enumerate(axes.flatten()):
        state = states[id]  # axes has same length as state
        ax.scatter(ref_data[0], ref_data[1][:, state-1, state-1], label='MD data')
        for n_tau in tau_data.keys():
            n_m_dt, T_to_m = tau_data[n_tau]
            ax.plot(n_m_dt, T_to_m[:, state-1, state-1], label='T^m('+n_tau+')')

        # ax.set_xscale(xscale)
        ax.set_title('State {}'.format(state))
    plt.legend()
    return fig, axes


def visualize_states(rect_dict, phi, psi, color_list):
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
    plt.text(2, 2, 'St1', color=colors['state1'], size=18)
    plt.text(-0.5, 2, 'St2', color=colors['state2'], size=18)
    plt.text(1, -.5, 'St3', color=colors['state3'], size=18)
    plt.show()
