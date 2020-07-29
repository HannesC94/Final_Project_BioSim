import numpy as np


def shrink_rec(rectangle, ds, where=['all']):
    rec = np.array(rectangle.copy())
    if len(rec.shape) == 3:
        print('ohauaha')
        for i, sub_rec in enumerate(rectangle):
            rectangle[i] = shrink_rec(sub_rec, ds=ds, where=where)
    # if not any([x == y]):
    #    print('where must be one of the following:\n"{}"\n"{}"\n"{}"\n"{}"\n"{}"'.format(
    #    *['l', 'r', 'b', 't', 'all']))
    if 'all' in where:
        rec[:, 0] += ds
        rec[:, 1] -= ds
    if 'l' in where:
        rec[0, 0] += ds
    if 'r' in where:
        rec[0, 1] -= ds
    if 'b' in where:
        rec[1, 0] += ds
    if 't' in where:
        rec[1, 1] -= ds
    return rec.tolist()


def change_core(rect_dict, ds):

    # divide ds by 2 to get a distance of ds between new rectangle sides
    ds = ds/2
    new_rec_dict = rect_dict.copy()
    state1_recs = new_rec_dict['state1']
    st1_ul = shrink_rec(state1_recs[0], ds, ['b', 'r'])
    st1_ur = shrink_rec(state1_recs[1], ds, ['b', 'l'])
    st1_ll = shrink_rec(state1_recs[2], ds, ['t', 'r'])
    st1_lr = shrink_rec(state1_recs[3], ds, ['t', 'l'])
    new_rec_dict['state1'] = [st1_ul, st1_ur, st1_ll, st1_lr]

    state2_recs = new_rec_dict['state2']
    st2_u = shrink_rec(state2_recs[0], ds, ['l', 'r', 'b'])
    st2_l = shrink_rec(state2_recs[1], ds, ['l', 'r', 't'])
    new_rec_dict['state2'] = [st2_u, st2_l]

    state3_recs = new_rec_dict['state3']
    st3 = shrink_rec(state3_recs[0], ds, ['b', 't'])
    new_rec_dict['state3'] = [st3]

    return new_rec_dict


# define rectangle(s) for state 1. put all lists in another list, called rec_list
s1_rec_ul = [[-np.pi, -2.07], [1.3, np.pi]]
s1_rec_ur = [[1.3, np.pi], [1.3, np.pi]]
s1_rec_ll = [[-np.pi, -2.07], [-np.pi, -2]]
s1_rec_lr = [[1.3, np.pi], [-np.pi, -2]]
rec_list_s1 = [s1_rec_ul, s1_rec_ur, s1_rec_ll, s1_rec_lr]

# define rectangle(s) for state 2. put all lists in another list, called rec_list
s2_rec_u = [[-2.07, 1.3], [1.3, np.pi]]
s2_rec_l = [[-2.07, 1.3], [-np.pi, -2]]
rec_list_s2 = [s2_rec_u, s2_rec_l]

# define rectangle(s) for state 3. put all lists in another list, called rec_list
s3_rec = [[-np.pi, np.pi], [-2, 1.3]]
rec_list_s3 = [s3_rec]

# make dictionary and save a dictionary. with rect_dict['statei'] you get a list
# which contains m lists with the phi and psi values, defining the rectangle. m
# is the number of rectangles, to which the state has been assigned.
rect_dict_1 = {
    'state1': rec_list_s1,
    'state2': rec_list_s2,
    'state3': rec_list_s3,
}

###################################################
# second defnition. Small angle between rectangles
# define angle width by which rectangles should be seperated in the Ramachandran
# plot
ds_2 = 5/360*2*np.pi  # 5degrees
rect_dict_2_core = change_core(rect_dict_1, ds=ds_2)

###################################################
# third defnition. Small angle between rectangles
# define angle width by which rectangles should be seperated in the Ramachandran
ds_3 = 30/360*2*np.pi  # 30degrees
rect_dict_3_core = change_core(rect_dict_1, ds=ds_3)
