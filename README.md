# ASSUDA
Code and data of [Exploring Robustness of Unsupervised Domain Adaptation in Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_Exploring_Robustness_of_Unsupervised_Domain_Adaptation_in_Semantic_Segmentation_ICCV_2021_paper.html) (ICCV 2021; Oral)


### Datasets

- [GTA5 -> Cityscapes (VGG)](https://drive.google.com/open?id=15XjJOuLHPinOu4FtYJunIoMQnHiMgpYc) and [GTA5 -> Cityscapes (DeepLab)](https://drive.google.com/open?id=1OBvYVz2ND4ipdfnkhSaseT8yu2ru5n5l)
- [SYNTHIA -> Cityscapes (VGG)](https://drive.google.com/open?id=1YlIHqLYTSL-JAGRLA8_9xDvOTnP3zVIs) and [SYNTHIA -> Cityscapes (DeepLab)](https://drive.google.com/open?id=1d7GxVhyN8HzEIPDeRIB3dRXTYzHI91ng)
- [Cityscapes](https://www.cityscapes-dataset.com/)

Perturbed test images in Cityscapes can be downloaded at: [PSPNet_Attack](https://drive.google.com/file/d/1iCNlxhlZLYRnyUuQll6JhST4YzBS2_hH/view?usp=sharing)

### Initial models
- [VGG](https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth)
- [DeepLab](https://drive.google.com/file/d/1TIrTmFKqEyf3pOKniv8-53m3v9SyBK0u/view)

### Training
```
python main.py \
    --data-dir /path/to/synthia_deeplab \
    --data-list ./dataset/synthia_list/train.txt \
    --data-dir-target /path/to/cityscapes \
    --data-list-target ./dataset/cityscapes_list/train.txt \
    --data-label-folder-target /path/to/synthia_deeplab/cityscapes_ssl \
    --snapshot-dir ./snapshots/synthia2city_deeplab \
    --init-weights ./initial_model/DeepLab_init.pth \
    --num-steps-stop 80000 \
    --model DeepLab \
    --source synthia \
    --learning-rate 1e-4 \
    --learning-rate-D 1e-6 \
    --lambda-adv-target 1e-4 \
    --save-pred-every 5000 \
    --alpha 1.0 \
    --lambda-contrastive 0.01
```

### Evaluation 
```
python evaluation.py \
    --data-dir-target /path/to/pspnet_attack/pspnet_fgsm_0.1 \
    --data-list-target ./dataset/cityscapes_list/val.txt \
    --gt_dir /path/to/cityscapes/gtFine/val \
    --devkit_dir ./dataset/cityscapes_list \
    --restore-from ./snapshots/synthia2city_deeplab/synthia_80000 \
    --save results/cityscapes_eval \
    --model DeepLab \
    --source synthia
```

### Citation

```
@article{yang2021exploring,
  title={Exploring Robustness of Unsupervised Domain Adaptation in Semantic Segmentation},
  author={Yang, Jinyu and Li, Chunyuan and An, Weizhi and Ma, Hehuan and Guo, Yuzhi and Rong, Yu and Zhao, Peilin and Huang, Junzhou},
  journal={Proceedings of the IEEE international conference on computer vision (ICCV)},
  year={2021}
}
```

### Acknowledgment
The code is heavily borrowed from [BDL](https://github.com/liyunsheng13/BDL)
