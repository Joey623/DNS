# [ECCV2024] [Domain Shifting: A Generalized Solution for Heterogeneous Cross-Modality Person Re-Identification](https://fq.pkwyx.com/default/https/www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09119.pdf).
Welcome to use the code from our paper "Domain Shifting:  A Generalized Solution for Heterogeneous Cross-Modality Person Re-Identification".

# Usage

* **Environment**

  ```
  torch >= 2.0.1
  ```

* **Dataset**

  ```
  Dataset/
  ├── SYSU-MM01
  ├── RegDB
  ├── LLCM
  DNS/
  ```

* **Preprocess the SYSU-MM01 and LLCM**

  ```
  python pre_process_sysu.py  # just for faster training
  python pre_process_llcm.py
  ```

* **Training**

  ```
  python train.py --dataset sysu --lr 0.2
  bash train_regdb.sh
  python train.py --dataset llcm --lr 0.2 
  ```

* **Testing**

  You can find the `training log` in [here](./log/) and the `ckpt` in [Google Drive](https://drive.google.com/file/d/18zdq4Ohit84h7khsnLq7MWUoMRQfajp6/view?usp=drive_link).  You can test DNS in [test.ipynb](./test.ipynb) or directly run the following command:

  ```
  # sysu all search
  python test.py --dataset sysu --resume sysu_p6_n4_lr_0.2_seed_0_best.pth --mode all
  
  # sysu indoor search
  python test.py --dataset sysu --resume sysu_p6_n4_lr_0.2_seed_0_best.pth --mode indoor
  
  # regdb infrared to visible
  python test.py --dataset regdb --tvsearch 0
  
  # regdb visible to infrared
  python test.py --dataset regdb --tvsearch 1
  
  # llcm infrared to visible
  python test.py --dataset llcm --resume llcm_p6_n4_lr_0.2_seed_0_best.pth --tvsearch 0
  
  # llcm visible to infrared
  python test.py --dataset llcm --resume llcm_p6_n4_lr_0.2_seed_0_best.pth --tvsearch 1
  ```

* **New Tutorial**

  I found that setting `args.max_epoch` to 80 has no impact on performance, but it can save 20% of training time. So if you wanna faster training, you can run the following command:

  ```
  python train.py --dataset sysu --lr 0.2 --max_epoch=80
  ```

  Please note that this is optional, and you can also keep in touch with me by setting `max_epoch` to 100. The corresponding log (`max_epoch=80` and `max_epoch=150`) is provided in [here](./log/sysu_p6_n4_lr_0.2_seed_0_epoch80&150.log). I hope this can help you.

# Acknowledgements

This code is built on [CAJ](https://github.com/mangye16/Cross-Modal-Re-ID-baseline/tree/master/ICCV21_CAJ), [DEEN](https://github.com/ZYK100/LLCM), and [ffcv-imagenet](https://github.com/libffcv/ffcv-imagenet) (DNS's training strategy), we thank the authors for opensourcing their code!

# Contact

If you have any questions, don't hesitate to contact me via [jiangyan@nuist.edu.cn](jiangyan@nuist.edu.cn).

# Citation

``` 
@inproceedings{jiang2024domain,
  title={Domain shifting: A generalized solution for heterogeneous cross-modality person re-identification},
  author={Jiang, Yan and Cheng, Xu and Yu, Hao and Liu, Xingyu and Chen, Haoyu and Zhao, Guoying},
  booktitle={European Conference on Computer Vision},
  pages={289--306},
  year={2024},
  organization={Springer}
}
```



