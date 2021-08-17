import os
import numpy as np
#python3 -m aligner -r eng.zip -a data/ -d eng.dict
#generate label for making pair of .wav/.lab for Prosodylab_Aligner
#filePath = args.textgrid_file



align_dir = 'data/Align_34_test/'
lab_dir = 'data/Lab_34_test/'
sub_name = os.listdir(align_dir)
sub_name.sort()
if os.path.exists(align_dir + '.DS_Store') is True:
    sub_name.remove('.DS_Store')
print(sub_name)



for sub_align in sub_name:
    os.makedirs(lab_dir + sub_align[:-5] + 'lab' + '/')
    sub_align_dir = align_dir + sub_align + '/' #Align/s1_align/
    print(sub_align)
    clip_name = os.listdir(sub_align_dir)
    clip_name.sort()
    if os.path.exists(sub_align_dir + '.DS_Store') is True:
        clip_name.remove('.DS_Store')
    #print(clip_name)
    for clip in clip_name:
        print(clip)
        f = open(sub_align_dir + clip)
        lines = f.read()
        print("ori_line", lines)
        print("ori_len", len(lines))
        num_space = lines.count(' ')
        print("len_spa", num_space)
        if lines.count(' ') == 16:
            lines=np.array(lines.split()).reshape(8,3)
        #if lines.count(' ') == 18:
        elif lines.count(' ') == 18:
            lines=np.array(lines.split()).reshape(9,3)
        elif lines.count(' ') == 20:
            lines=np.array(lines.split()).reshape(10,3)
        elif lines.count(' ') == 22:
            lines=np.array(lines.split()).reshape(11,3)
        elif lines.count(' ') == 24:
            lines=np.array(lines.split()).reshape(12,3)

        print("**",lines)
        print(np.shape(lines))
        label= lines[:,2]
        label = label[1:-1]
        for i in range(len(label)):
            label[i] =  label[i].upper()
        print(label)
        print(len(label))
        if len(label) != 6 and len(label) !=7 and len(label) != 8 and len(label) !=9 and len(label) !=10:
            print("ERROR!!!!!!!!!!!!!!!!!!")
            break


        np.savetxt(lab_dir + sub_align[:-5] + 'lab' + '/' + clip[:-6] + ".lab", label, fmt='%s', newline=' ')