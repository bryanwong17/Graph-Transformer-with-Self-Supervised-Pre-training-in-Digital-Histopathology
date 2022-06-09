import os
from datetime import datetime

path = 'E:/sample_3000/val/LUAD'
total_slide = len(os.listdir(path))
i = 0

for index, slide in enumerate(os.listdir(path)):
    for idx, patch in enumerate(os.listdir(os.path.join(path,slide))):
        old_path = os.path.join(path,slide,patch)
        new_path = os.path.join(path,slide,f'{slide}-{patch}')
        # name = patch.split('_')
        # new_path = os.path.join(path,slide,f'{slide}-{name[1]}_{name[2]}')
        os.rename(old_path,new_path)
    i += 1
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'[{i}/{total_slide}] {current_time}')