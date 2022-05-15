import os
import shutil

dirnames = os.listdir('../../dataset/Raw/RGB (320 x 240)/')
dirnames.sort() # order by filename
dirnames = dirnames[1:]
indices = []
labels = []
for dname in dirnames:
    for fl in os.listdir('../../dataset/Raw/RGB (320 x 240)/'+dname+'/'):
        shutil.move('../../dataset/Raw/RGB (320 x 240)/'+dname+'/'+fl,'../../dataset/Raw/Combined/')