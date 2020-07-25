import numpy as np

# function to shrink rectangles


def shrink_rec(rec, ds):
    rec = np.array(rec)
    if len(rec.shape) == 2:
        rec[:, 0] += ds
        rec[:, 1] -= ds
    elif len(rec.shape) == 3:
        rec[:, :, 0] += ds
        rec[:, :, 1] -= ds
    return rec.tolist()


# define rectangle(s) for state 1. put all lists in another list, called rec_list
s1_rec_ul = [[-np.pi, -2.07], [1.3, np.pi]]
s1_rec_ur = [[1.3, np.pi], [1.3, np.pi]]
s1_rec_ll = [[-np.pi, -2.07], [-np.pi, -2]]
s1_rec_lr = [[1.3, np.pi], [-np.pi, -2]]
rec_list_s1 = [s1_rec_ul, s1_rec_ur, s1_rec_ll, s1_rec_lr]

# define rectangle(s) for state 2. put all lists in another list, called rec_list
s2_rec = [[-2.07, 1.3], [-np.pi, -2]]
s2_rec_2 = [[-2.07, 1.3], [1.3, np.pi]]
rec_list_s2 = [s2_rec, s2_rec_2]

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

##################################################
# second defnition. Small angle between rectangles

# define angle width by which rectangles should be seperated in the Ramachandran
# plot
ds_2 = 5/360*2*np.pi  # 5degrees
# make copy of unseperated rectangle dict
rect_dict_2 = rect_dict_1.copy()
for state in rect_dict_2.keys():
    rect_dict_2[state] = shrink_rec(rect_dict_2[state], ds=ds_2)

##################################################
# third defnition. Small angle between rectangles
# define angle width by which rectangles should be seperated in the Ramachandran
ds_3 = 30/360*2*np.pi  # 30degrees
# make copy of unseperated rectangle dict
rect_dict_3 = rect_dict_1.copy()
for state in rect_dict_3.keys():
    rect_dict_3[state] = shrink_rec(rect_dict_3[state], ds=ds_3)
