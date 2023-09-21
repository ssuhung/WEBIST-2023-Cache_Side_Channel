import argparse
import multiprocessing
import os
from multiprocessing import Pool

import numpy as np
import progressbar

# Parameters (Set these by yourself)
repo_root = '../'
target_path = os.path.join(repo_root, 'target/tjexample')
npz_output_root = os.path.join(repo_root, 'data/CelebA_jpg/pin/raw/')
img_root = os.path.join(repo_root, 'data/CelebA_jpg/image/')
splits = ['train/', 'test/']
# widgets = ['Progress: ', progressbar.Percentage(), ' ', 
#             progressbar.Bar('#'), ' ', 'Count: ', progressbar.Counter(), ' ',
#             progressbar.Timer(), ' ', progressbar.ETA()]

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def raw2npz(in_path, npz_path) -> int:
    with open(in_path, 'r') as f:
        lines = f.readlines()

    mem_arr = []
    for info in lines[:-1]:
        # Format: "lib;rtn;ins op addr"
        op, addr_16 = info.split(' ')[-2:]
        addr = int(addr_16, 16)
        if op == 'R':
            mem_arr.append(addr)
        elif op == 'W':
            mem_arr.append(-addr)
        else:
            print(f'Unknown operation in {in_path}')

    # print('Length: ', len(mem_arr))
    # if len(mem_arr) < pad_length:
    #     mem_arr += [0] * (pad_length - len(mem_arr))
    # else:
    #     mem_arr = mem_arr[:pad_length]

    np.savez_compressed(npz_path, np.array(mem_arr))
    return len(mem_arr)

def collect(img, image_dir, split):
    worker_id = multiprocessing.current_process().name.split('-')[-1]
    pin_out = f'mem_access_{worker_id}.out'
    pin = f'../../../pin -t obj-intel64/mem_access.so -o {pin_out}'

    img_path = os.path.join(image_dir, img)
    prefix = img.split('.')[0]
    npz_path = os.path.join(npz_output_root, split, prefix + '.npz')
    
    # Target libjpeg
    os.system(f'{pin} -- {target_path} {img_path} img_output_{worker_id}.bmp > /dev/null')
    # Target libwebp
    # os.system(f'{pin} -- {target_path} {img_path} -bmp -o img_output_{ID}.bmp > /dev/null 2>&1')

    leng = raw2npz(in_path=pin_out, npz_path=npz_path)
    if int(prefix) % 1000 == 0:
        print(f'Files before id {prefix} is done')
    return leng

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers to paralize')
    args = parser.parse_args()

    make_path(npz_output_root)
    max_len = 0

    for split in splits:
        image_dir = os.path.join(img_root, split)
        img_list = sorted(os.listdir(image_dir))
        # progress = progressbar.ProgressBar(widgets=widgets, maxval=len(img_list)).start()

        make_path(os.path.join(npz_output_root, split))

        print('Total number of images: ', len(img_list))

        img_dirs = [image_dir] * len(img_list)
        split_list = [split] * len(img_list)
        inputs = list(zip(img_list, img_dirs, split_list))

        with Pool(args.num_workers) as p:
            result = p.starmap(collect, inputs)
            max_len = max(max(result), max_len)
            # progress.finish()

    print(f"{max_len=}")
