import os
import shutil

# dirnames = os.listdir('../../dataset/Raw/RGB (320 x 240)/')
dirnames = os.listdir('../../dataset/CORF3D')
dirnames.sort() # order by filename
dirnames = dirnames[1:]
indices = []
labels = []
total = 0
for dname in dirnames:
    count = 0
    for fl in os.listdir('../../dataset/CORF3D/'+dname+'/'):
        count += 1
    # if count == 2:
    #     total += 1
    #     print(dname)
        # shutil.move('../../dataset/Raw/RGB (320 x 240)/'+dname+'/'+fl,'../../dataset/Raw/Combined/')
        shutil.copy('../../dataset/CORF3D/'+dname+'/'+fl,'../../dataset/CORF3D_combined/')
print(total)