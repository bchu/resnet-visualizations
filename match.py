import os

f1 = './caffenet_conv345/conv4'
f2 = './caffenet_conv345/conv5'

def create_sets(foldername):
    subdirs = next(os.walk(foldername))[1]
    return {subdir: get_max_ims(os.path.join(foldername, subdir)) for subdir in subdirs}
    
def get_max_ims(unit_dir):
    info_fname = os.path.join(unit_dir, 'info.txt')
    ims = set()
    with open(info_fname) as info_file:
        info_file.readline()
        for line in info_file:
            line = line[:-1]
            max_im = line.split(' ')[6]
            ims.add(max_im)
    return ims

layer_4 = create_sets(f1)
layer_5 = create_sets(f2)

corrs = []
for l4_unit, l4_maxes in layer_4.items():
    for l5_unit, l5_maxes in layer_5.items():
        corrs.append((len(l4_maxes & l5_maxes), l4_unit, l5_unit))

corrs = sorted(corrs)[::-1]

for corr in corrs[:10]:
    print(corr, end='')
    os.system('open -a Preview {}/{}/*.png'.format(f1, corr[1]))
    os.system('open -a Preview {}/{}/*.png'.format(f2, corr[2]))
    input()

