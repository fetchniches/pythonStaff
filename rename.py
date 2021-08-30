import os
import glob
import shutil
import re

def rename(src: os.PathLike, dst: os.PathLike, offset: int, suffix: str = 'jpg'):
    files_name = glob.glob(os.path.join(src, '*.'+suffix))
    success: int = 0
    for fn in files_name:
        try:
            fname = str(int(re.search(r'\d*\.'+suffix+'$', fn).group()[:-4])+offset)+'.'+suffix
            to_fn = os.path.join(dst, fname)
            if os.path.exists(to_fn):
                raise FileExistsError('file alread exists! Exit rename.')
            shutil.copy(fn, to_fn)
            print(fn, '->', to_fn)
            success += 1
        except Exception as e:
            print('File {} goes wrong, caused by {}'.format(fn, e))
    print('{} file(s) renamed successfully, {} file(s) failed.'.format(success, len(files_name)-success))

if __name__ == '__main__':
    rename('./imagesAndLabels/Box2Label', 2, 'txt')