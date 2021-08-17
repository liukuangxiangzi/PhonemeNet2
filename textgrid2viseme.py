import textgrid
import os
import numpy as np
import argparse

#python textgrid2viseme.py -t path-of-textgrid-file, -o path-of-saving-lab-viseme-file
#python textgrid2viseme.py -t Grid/s2_mfa -o viseme_fold/s2

# parser = argparse.ArgumentParser(description=__doc__)
# parser.add_argument("-t", "--textgrid_file", type=str, help="input Textgrid file")
# parser.add_argument("-o", "--viseme_file", type=str, help="save viseme label")
# args = parser.parse_args()

fps = 25
number_of_frame = 75
frames = np.arange(0, number_of_frame)
seconds_per_frame = frames/25

phoneme_dict = {
    'ER0':9, 'AA1': 9, 'AE1': 9, 'AE2': 9, 'AH0': 9, 'AH1': 9, 'AO1': 11, 'AW1': 9, 'AY1': 9, 'B': 0, 'SH': 6, 'CH': 6, 'D': 5, 'DH': 3,
    'EH1': 9, 'EY1': 9, 'IY2': 9,'F': 2, 'G': 7, 'HH': 8, 'IH0': 9, 'IH1': 9, 'IY0': 9,'IY1': 9, 'NG':9, 'JH': 6, 'K': 7, 'L': 4,
    'M': 0, 'N': 5, 'OW0': 11, 'OW1': 11, 'OW2': 11, 'P': 0, 'R': 5, 'S': 6, 'T': 5, 'TH': 3, 'UW1': 10, 'V': 2, 'W': 1,
    'Y': 7, 'Z': 5, 'sil': 12, 'sp': 12
}



all_grid_dir = 'test_grid/'            #args.textgrid_file
grid_subs = os.listdir(all_grid_dir)
grid_subs.sort()
if os.path.exists(all_grid_dir + '.DS_Store') is True:
    grid_subs.remove('.DS_Store')
print("****", len(grid_subs))
sub_name_list = []
c_sub = 0
viseme = []
for a_grid_sub in grid_subs:  #34
    sub_name_list.append(a_grid_sub)
    c_sub += 1
    print('c_sub', c_sub)
    #'a_grid_sub_dir' is 'filePath'
    a_grid_sub_dir = all_grid_dir + a_grid_sub + '/'
    grid_seqs = os.listdir(a_grid_sub_dir) #1000
    #'grid_seqs' is 'names'
    grid_seqs.sort()
    if os.path.exists(a_grid_sub_dir + '.DS_Store') is True:
        grid_seqs.remove('.DS_Store')
    # if len(grid_seqs) != 1000:
    #     print('num seqs in'+ a_grid_sub + 'is not 1000')
    #     break

    c_seq = 0
    all_frame_label = []
    for n in range(len(grid_seqs)):
        # # Read a TextGrid object from a file.
        tg = textgrid.TextGrid.fromFile(a_grid_sub_dir + grid_seqs[n])
        # # Read a IntervalTier object.

        c_seq += 1
        count = 0
        for t in seconds_per_frame:
            count+=1
            for i in range(len(tg[1][:])):
                if tg[1][i].minTime <= t <tg[1][i].maxTime:
                    all_frame_label.append(phoneme_dict[tg[1][i].mark])
    # print("^allframelable", len(all_frame_label))
        #print("processed_clip_num", c_seq)
    # all_frame_label = np.array(all_frame_label)
    # all_frame_label = all_frame_label.reshape(c_seq, number_of_frame)

    viseme.append(all_frame_label)
viseme = np.array(viseme)
print("*shape", viseme.shape)
viseme = viseme.reshape(-1,number_of_frame)

print('v_all', viseme.shape)
#print('*', viseme[1000])
np.save('test_grid/viseme_test_s2_bbim3a-5', viseme)#test:191*65 train:753*75  #viseme_all
print('sub_name', sub_name_list)
#np.savetxt('data/name_list_order_viseme_lab.txt', sub_name_list, fmt='%s')