# TSD

## Related information
Here is the code of "Learning a Task-specific Descriptor for Robust Matching of 3D Point Clouds" (``https://ieeexplore.ieee.org/abstract/document/9847261``), which proposes to learn a matching task-specific feature descriptor.

<!--Note: the code is being prepared. -->

## Implementation
The code is tested with Pytorch 1.6.0 with CUDA 10.2.89. Prerequisites include scipy, h5py, tqdm, etc. Your can install them by yourself.

Compile the C++ extension module for python located in cpp_wrappers. Open a terminal in this folder, and run:
```
sh compile_wrappers.sh
```


The 3DMatch and KITTI datasets can be download from:
```
https://github.com/XuyangBai/D3Feat
```

Start training with the command:
```
python train.py 
```

Start testing with the command:
```
python test.py --chosen_snapshot
```

## Acknowledgement
The code is insipred by D3Feat, KPConv, etc.

## Please cite:
```
@ARTICLE{zhang_tsd_tcsvt_2022,
  title={Learning a Task-specific Descriptor for Robust Matching of 3D Point Clouds},
  author={Zhang, Zhiyuan and Dai, Yuchao and Fan, Bin and Sun, Jiadai and He, Mingyi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022},
  volume={32},
  number={12},
  pages={8462-8475}} 
```
