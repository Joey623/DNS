# Domain Shifting: A Generalized Solution for Heterogeneous Cross-Modality Person Re-Identification.
## Abstract

Cross-modality person re-identification (ReID) is a challenging task that aims to match cross-modality pedestrian images across multiple camera views. Existing methods are tailored to specific tasks and perform well for visible-infrared or visible-sketch ReID. However, the performance exhibits a notable decline when the same method is utilized for multiple cross-modality ReIDs, limiting its generalization and applicability. To address this issue, we propose a generalized domain shifting method (DNS) for cross-modality ReID, which can address the generalization and perform well in both visible-infrared and visible-sketch modalities. Specifically, we propose the heterogeneous space shifting and common space shifting modules to augment specific and shared representations in heterogeneous space and common space, respectively, thereby regulating the model to learn the consistency between modalities. Further, a domain alignment loss is developed to alleviate the cross-modality discrepancies by aligning the patterns across modalities. In addition, a domain distillation loss is designed to distill identity-invariant knowledge by learning the distribution of different modalities. Extensive experiments on two cross-modality ReID tasks (*i.e.*, visible-infrared ReID, visible-sketch ReID) demonstrate that the proposed method outperforms the state-of-the-art methods by a large margin.



## Training

Before training, you need preprocess the SYSU-MM01 and LLCM as follow:

```
python pre_process_sysu.py
python pre_process_llcm.py
```

You can run the following command to train：

```
python train.py --dataset sysu --lr 0.2 # train sysu
bash train_regdb.sh # train regdb
python train.py --dataset llcm --lr 0.2 # train llcm
```

## Testing

You can find the checkpoint and the command in the [test.ipynb](./test.ipynb). 





