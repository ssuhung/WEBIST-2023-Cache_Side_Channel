import numpy as np
import os
import time

dir_name = '../data/CelebA_WebP/pin/raw/test/'
output_dir = '../data/CelebA_WebP/pin/raw_uncomp/test/'
output_type = np.int16

def to_cacheline(trace):
    return np.where(trace > 0, abs(trace) & 0xFFF, -(abs(trace) & 0xFFF))

if __name__ == '__main__':
    file_list = sorted(os.listdir(dir_name))
    max_len = 0
    start_time = time.time()

    for file in file_list:
        trace = np.load(dir_name + file)['arr_0']
        trace = to_cacheline(trace).astype(output_type)
        np.savez(output_dir + file, trace)

    print(f'Time consumption {time.time() - start_time}s')
