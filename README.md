# Cache Side-Channel Attacks Against Black-Box Image Processing Software
This is the artifect of the cache side-channel analysis framework proposed in WEBIST 2023 paper:  [Cache Side-Channel Attacks Against Black-Box Image Processing Software]()

# 0. Install Dependencies
Installing python dependency by running

```shell
pip install -r requirements.txt
```

# 1. Prepare Image Dataset
1. Download the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), Align & Cropped version.
2. Set the `input_dir` variable in the `util/crop_celeba.py` to the path of downloaded dataset
3. Execute the script to crop images

```shell
cd util
python3 crop_and_webp.py
```

# 2. Prepare Victim Programs
To prepare the victom image processing programs: 
## JPEG
Follow the instruction in [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) repository. Install the library and compile `tjexample.c`, then move the compiled executable into `target/` folder
## WebP
Follow the instruction in [libwebp](https://github.com/webmproject/libwebp) repository. Install the library and compile `dwebp.c`, then move the compiled executable into `target/` folder

# 3. Collect Traces
1. Download [Intel Pin](https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-dynamic-binary-instrumentation-tool.html) and unzip it
2. Copy `pin/pintool/mem_access.cpp` into `<pin root>/source/tools/ManualExamples/` and run
```shell
make obj-intel64/mem_access.so TARGET=intel64
```

3. Copy `pin/prep_traces.py` to `<pin root>/source/tools/ManualExamples/` and set the `repo_root` variable to the directory of this repository in the file:

4. Execute the script to collect traces. Use `--num_workers` option to accelerate the process
```shell
cd <pin root>/source/tools/ManualExamples/
python3 prep_traces.py　--num_workers <number of workers>
```
(Optional) 5. To train the model with traces collected from `libwebp`, uncompressing the npz file and preforming pre-processing is necessary for optimizing the proformance. Execute the `util/uncompress_npz.py` script

# 4. Train and Evaluate
Train the neural network model by the following command
```shell
cd src
python3 recons_image.py --side <cacheline or cacheline_encode> --dataset <CelebA_jpg or CelebA_webp> --attack <pp or wb> --exp_name <arbitrary_experiment_name>
```
The model, samples of reference images and reconstructed images will be saved in `output/<arbitrary_experiment_name>`

# Common Problem

1.  Shared Memory Not Enough or Worker of Dataloader Terminate Unexpectedly

    Try to decrease the number of workers with parameter `--num_workers`