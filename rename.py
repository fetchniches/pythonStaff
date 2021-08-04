import os
import glob
import re

def rename(folder: os.PathLike, offset: int, suffix: str = 'jpg'):
    files_name = glob.glob(os.path.join(folder, '*.'+suffix))
    success: int = 0
    for fn in files_name:
        try:
            dst = str(int(re.search(r'\d*\.'+suffix+'$', fn).group()[:-4])+offset)+'.'+suffix
            dst = os.path.join(folder, dst)
            os.rename(fn, dst)
            print(fn, '->', dst)
            success += 1
        except Exception as e:
            print('File {} goes wrong, caused by {}'.format(fn, e))
    print('{} file(s) renamed successfully, {} file(s) failed.'.format(success, len(files_name)-success))

if __name__ == '__main__':
    rename('./imagesAndLabels/Box2Label', 2, 'txt')