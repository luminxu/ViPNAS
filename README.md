# ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search

\[[paper](https://arxiv.org/abs/2105.10154)\]

## Introduction

This is the official implementation of
[*ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search*](https://arxiv.org/abs/2105.10154) (CVPR'2021) paper.

Human pose estimation has achieved significant progress in recent years. However, most of the recent methods focus on improving accuracy using complicated models and ignoring real-time efficiency. To achieve a better trade-off between accuracy and efficiency, we propose a novel neural architecture search (NAS) method, termed ViPNAS, to search networks in both spatial and temporal levels for fast online video pose estimation. In the spatial level, we carefully design the search space with five different dimensions including network depth, width, kernel size, group number, and attentions. In the temporal level, we search from a series of temporal feature fusions to optimize the total accuracy and speed across multiple video frames. To the best of our knowledge, we are the first to search for the temporal feature fusion and automatic computation allocation in videos. Extensive experiments demonstrate the effectiveness of our approach on the challenging COCO2017 and PoseTrack2018 datasets. Our discovered model family, S-ViPNAS and T-ViPNAS, achieve significantly higher inference speed (CPU real-time) without sacrificing the accuracy compared to the previous state-of-the-art methods.

Our code is reimplemented based on [MMPose](https://github.com/open-mmlab/mmpose).

## Results and models

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | ImageNet Pretrain | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | AR | ckpt | log |
| :-------------- | :-----------: |:-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |:------: |
| ResNet-50 (paper)  | 256x192 | Y | 0.704 | 0.886 | 0.783 | 0.671 | 0.772 | 0.763 |  |  |
| S-VipNAS-Res50 (paper)  | 256x192 | N | 0.710 | 0.893 | 0.787 | 0.677 | 0.775 | 0.767 |  |  |
| [S-VipNAS-Res50](https://github.com/luminxu/ViPNAS/blob/main/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/s_vipnas_res50_coco_256x192.py)  | 256x192 | N | 0.711 | 0.893 | 0.789 | 0.679 | 0.777 | 0.769 | [ckpt](https://drive.google.com/file/d/1maPJb8bTsUZOEljp-ceSvJFz3Jyn3VbO/view) | [log](https://drive.google.com/file/d/1VRmsvPKvpaPqHqVZYqmX1aatGzfmCWRS/view) |

## Enviroment

The codes are developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 8 NVIDIA V100 GPU cards. Other platforms or GPU cards are not fully tested.

## Quick Start

### Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv) (Please install the latest version of mmcv-full)
- Numpy
- cv2
- json_tricks
- [xtcocotools](https://github.com/jin-s13/xtcocoapi)

### Installation

<!-- The code is based on [MMPose](https://github.com/open-mmlab/mmpose).
You need clone the mmpose project and integrate the codes into mmpose first. -->

a. Install mmcv, we recommend you to install the pre-build mmcv as below.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please replace ``{cu_version}`` and ``{torch_version}`` in the url to your desired one. For example, to install the latest ``mmcv-full`` with ``CUDA 11`` and ``PyTorch 1.7.0``, use the following command:

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

If it compiles during installation, then please check that the cuda version and pytorch version **exactly"" matches the version in the mmcv-full installation command. For example, pytorch 1.7.0 and 1.7.1 are treated differently.
See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

Optionally you can choose to compile mmcv from source by the following command

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full, which contains cuda ops, will be installed after this step
# OR pip install -e .  # package mmcv, which contains no cuda ops, will be installed after this step
cd ..
```

Or directly run

```shell
pip install mmcv-full
# alternative: pip install mmcv
```

**Important:** You need to run `pip uninstall mmcv` first if you have mmcv installed. If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

b. Install build requirements

```shell
pip install -r requirements.txt
```

### Prepare datasets

It is recommended to symlink the dataset root to `MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. [HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation) provides person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-)
Download and extract them under `MMPOSE/data`, and make them look like this:

```
mmpose
├── configs
├── models
├── tools
`── data
    │── coco
        │-- annotations
        │   │-- person_keypoints_train2017.json
        │   |-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        │-- train2017
        │   │-- 000000000009.jpg
        │   │-- 000000000025.jpg
        │   │-- 000000000030.jpg
        │   │-- ...
        `-- val2017
            │-- 000000000139.jpg
            │-- 000000000285.jpg
            │-- 000000000632.jpg
            │-- ...

```

## Training and Testing

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by modifying the interval argument in the training config

```python
evaluation = dict(interval=5)  # This evaluate the model per 5 epoch.
```

According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or videos per GPU, e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.

### Training

```shell
# train with a signle GPU
python tools/train.py ${CONFIG_FILE} [optional arguments]

# train with multiple GPUs
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--validate` (**strongly recommended**): Perform evaluation at every k (default value is 5 epochs during the training.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--gpus ${GPU_NUM}`: Number of gpus to use, which is only applicable to non-distributed training.
- `--seed ${SEED}`: Seed id for random state in python, numpy and pytorch to generate random numbers.
- `--deterministic`: If specified, it will set deterministic options for CUDNN backend.
- `JOB_LAUNCHER`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `LOCAL_RANK`: ID for local rank. If not specified, it will be set to 0.
- `--autoscale-lr`: If specified, it will automatically scale lr with the number of gpus by [Linear Scaling Rule](https://arxiv.org/abs/1706.02677).

Difference between `resume-from` and `load-from`:
`resume-from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load-from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

Examples:

#### Training on COCO train2017 dataset

```shell
./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/s_vipnas_res50_coco_256x192.py 8
```

### Testing

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRIC}] \
    [--proc_per_gpu ${NUM_PROC_PER_GPU}] [--gpu_collect] [--tmpdir ${TMPDIR}] [--average_clips ${AVG_TYPE}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]

# multiple-gpu testing
./tools/dist_test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRIC}] \
    [--proc_per_gpu ${NUM_PROC_PER_GPU}] [--gpu_collect] [--tmpdir ${TMPDIR}] [--average_clips ${AVG_TYPE}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]
```

Optional arguments:

- `RESULT_FILE`: Filename of the output results. If not specified, the results will not be saved to a file.
- `EVAL_METRIC`: Items to be evaluated on the results. Allowed values depend on the dataset.
- `NUM_PROC_PER_GPU`: Number of processes per GPU. If not specified, only one process will be assigned for a single gpu.
- `--gpu_collect`: If specified, recognition results will be collected using gpu communication. Otherwise, it will save the results on different gpus to `TMPDIR` and collect them by the rank 0 worker.
- `TMPDIR`: Temporary directory used for collecting results from multiple workers, available when `--gpu_collect` is not specified.
- `AVG_TYPE`: Items to average the test clips. If set to `prob`, it will apply softmax before averaging the clip scores. Otherwise, it will directly average the clip scores.
- `JOB_LAUNCHER`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `LOCAL_RANK`: ID for local rank. If not specified, it will be set to 0.

Examples:

#### Test S-ViPNAS-Res50 on COCO with 8 GPUS, and evaluate the mAP

```shell
./tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/s_vipnas_res50_coco_256x192.py \
    checkpoints/SOME_CHECKPOINT.pth 8 \
    --eval mAP
```

## Acknowledgement

Thanks to:

- [MMPose](https://github.com/open-mmlab/mmpose)

## Citations

Please consider citing our paper in your publications, if the project helps your research. BibTeX reference is as follows.

```
@article{xu2021vipnas,
  title={ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search},
  author={Xu, Lumin and Guan, Yingda and Jin, Sheng and Liu, Wentao and Qian, Chen and Luo, Ping and Ouyang, Wanli and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2021}
}
```

## License

Our research code is released under the MIT license. Please see the [LICENSE](LICENSE) for further details.
