# Complex YOLOv4 Reproducing Project for Computer Graphics

This project is based on the original PyTorch Implementation of Complex-YOLOv4.

---

## Folder structure

```
${ROOT}
└── checkpoints/    
    ├── complex_yolov3/
    └── complex_yolov4/
└── dataset/    
    └── kitti/
        ├──ImageSets/
        │   ├── train.txt
        │   └── val.txt
        ├── training/
        │   ├── image_2/ <-- for visualization
        │   ├── calib/
        │   ├── label_2/
        │   └── velodyne/
        └── testing/  
        │   ├── image_2/ <-- for visualization
        │   ├── calib/
        │   └── velodyne/ 
        └── classes_names.txt
└── src/
    ├── config/
    ├── cfg/
        │   ├── complex_yolov3.cfg
        │   ├── complex_yolov3_tiny.cfg
        │   ├── complex_yolov4.cfg
        │   ├── complex_yolov4_tiny.cfg
    │   ├── train_config.py
    │   └── kitti_config.py
    ├── data_process/
    │   ├── kitti_bev_utils.py
    │   ├── kitti_dataloader.py
    │   ├── kitti_dataset.py
    │   ├── kitti_data_utils.py
    │   ├── train_val_split.py
    │   └── transformation.py
    ├── models/
    │   ├── darknet2pytorch.py
    │   ├── darknet_utils.py
    │   ├── model_utils.py
    │   ├── yolo_layer.py
    └── utils/
    │   ├── evaluation_utils.py
    │   ├── iou_utils.py
    │   ├── logger.py
    │   ├── misc.py
    │   ├── torch_utils.py
    │   ├── train_utils.py
    │   └── visualization_utils.py
    ├── evaluate.py
    ├── test.py
    ├── test.sh
    ├── train.py
    └── train.sh
├── README.md 
└── requirements.txt
```

## Usage

```
usage: train.py [-h] [--seed SEED] [--saved_fn FN] [--working-dir PATH]
                [-a ARCH] [--cfgfile PATH] [--pretrained_path PATH]
                [--img_size IMG_SIZE] [--hflip_prob HFLIP_PROB]
                [--cutout_prob CUTOUT_PROB] [--cutout_nholes CUTOUT_NHOLES]
                [--cutout_ratio CUTOUT_RATIO]
                [--cutout_fill_value CUTOUT_FILL_VALUE]
                [--multiscale_training] [--mosaic] [--random-padding]
                [--no-val] [--num_samples NUM_SAMPLES]
                [--num_workers NUM_WORKERS] [--batch_size BATCH_SIZE]
                [--print_freq N] [--tensorboard_freq N] [--checkpoint_freq N]
                [--start_epoch N] [--num_epochs N] [--lr_type LR_TYPE]
                [--lr LR] [--minimum_lr MIN_LR] [--momentum M] [-wd WD]
                [--optimizer_type OPTIMIZER] [--burn_in N]
                [--steps [STEPS [STEPS ...]]] [--world-size N] [--rank N]
                [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
                [--gpu_idx GPU_IDX] [--no_cuda]
                [--multiprocessing-distributed] [--evaluate]
                [--resume_path PATH] [--conf-thresh CONF_THRESH]
                [--nms-thresh NMS_THRESH] [--iou-thresh IOU_THRESH]

The Implementation of Complex YOLOv4

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           re-produce the results with seed random
  --saved_fn FN         The name using for saving logs, models,...
  --working-dir PATH    The ROOT working directory
  -a ARCH, --arch ARCH  The name of the model architecture
  --cfgfile PATH        The path for cfgfile (only for darknet)
  --pretrained_path PATH
                        the path of the pretrained checkpoint
  --img_size IMG_SIZE   the size of input image
  --hflip_prob HFLIP_PROB
                        The probability of horizontal flip
  --cutout_prob CUTOUT_PROB
                        The probability of cutout augmentation
  --cutout_nholes CUTOUT_NHOLES
                        The number of cutout area
  --cutout_ratio CUTOUT_RATIO
                        The max ratio of the cutout area
  --cutout_fill_value CUTOUT_FILL_VALUE
                        The fill value in the cut out area, default 0. (black)
  --multiscale_training
                        If true, use scaling data for training
  --mosaic              If true, compose training samples as mosaics
  --random-padding      If true, random padding if using mosaic augmentation
  --no-val              If true, dont evaluate the model on the val set
  --num_samples NUM_SAMPLES
                        Take a subset of the dataset to run and debug
  --num_workers NUM_WORKERS
                        Number of threads for loading data
  --batch_size BATCH_SIZE
                        mini-batch size (default: 4), this is the totalbatch
                        size of all GPUs on the current node when usingData
                        Parallel or Distributed Data Parallel
  --print_freq N        print frequency (default: 50)
  --tensorboard_freq N  frequency of saving tensorboard (default: 20)
  --checkpoint_freq N   frequency of saving checkpoints (default: 2)
  --start_epoch N       the starting epoch
  --num_epochs N        number of total epochs to run
  --lr_type LR_TYPE     the type of learning rate scheduler (cosin or
                        multi_step)
  --lr LR               initial learning rate
  --minimum_lr MIN_LR   minimum learning rate during training
  --momentum M          momentum
  -wd WD, --weight_decay WD
                        weight decay (default: 1e-6)
  --optimizer_type OPTIMIZER
                        the type of optimizer, it can be sgd or adam
  --burn_in N           number of burn in step
  --steps [STEPS [STEPS ...]]
                        number of burn in step
  --world-size N        number of nodes for distributed training
  --rank N              node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --gpu_idx GPU_IDX     GPU index to use.
  --no_cuda             If true, cuda is not used.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
  --evaluate            only evaluate the model, not training
  --resume_path PATH    the path of the resumed checkpoint
  --conf-thresh CONF_THRESH
                        for evaluation - the threshold for class conf
  --nms-thresh NMS_THRESH
                        for evaluation - the threshold for nms
  --iou-thresh IOU_THRESH
                        for evaluation - the threshold for IoU
```

[python-image]: https://img.shields.io/badge/Python-3.6-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.5-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
