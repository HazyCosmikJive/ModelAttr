# 🔬 ModelAttr - Model Attribution Codebase

A simple codebase for Model Attribution task based on PyTorch.

Besides model attribution, these models can be modified from multi-class classification to binary classification for  generated image detection task.

## 📦 Supported Methods

| Method    | Paper                                                                                                      | Links                                                                                                                                                                                                                         |
| --------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AttNet    | Attributing fake images to gans: Learning and analyzing gan fingerprints. (ICCV 2019)                      | [paper ](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Attributing_Fake_Images_to_GANs_Learning_and_Analyzing_GAN_Fingerprints_ICCV_2019_paper.pdf)[code](https://github.com/ningyu1991/GANFingerprints)               |
| LeveFreq  | Leveraging frequency analysis for deep fake image recognition. (ICML 2020)                                 | [paper](https://arxiv.org/pdf/2003.08685.pdf) [code](https://github.com/RUB-SysSec/GANDCTAnalysi)                                                                                                                                   |
| DNA-Det   | Deepfake network architecture attribution. (AAAI 2022)                                                     | [paper](https://arxiv.org/pdf/2202.13843.pdf) [code](https://github.com/ICTMCG/DNA-Det)                                                                                                                                            |
| PatchCNN* | What Makes Fake Images Detectable? Understanding Properties that Generalize. (ECCV 2020))                  | [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710103.pdf) [code](https://github.com/chail/patch-forensics)                                                                                                   |
| UniFD*    | Towards universal fake image detectors that generalize across generative models (CVPR 2023)                | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.pd) [code](https://github.com/Yuheng-Li/UniversalFakeDetec)      |
| LGrad*    | Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection. (CVPR2023) | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tan_Learning_on_Gradients_Generalized_Artifacts_Representation_for_GAN-Generated_Images_Detection_CVPR_2023_paper.pd) [code](https://github.com/chuangchuangtan/LGra) |

Note that methods with the mark `*` are originally designed for generated image detection task.

In folder `exps/baselines`, we provide the config for training and testing these methods.

## 🛠 Requirements

- Linux
- NVIDIA GPU + CUDA 11.1
- Python == 3.8.5
- PyTorch == 1.10.0+cu111

All dependencies are listed in `requirements.txt`, you may run `pip install -r requirements.txt` to use the same experimental settings. Other versions of libs may also work.

## 🤔 Usage

Here we take the simplest resnet18 experiments in the `baselines` folder for example.

Please refer to the comments in `exps/baselines/01_resnet18.py` for more details about config parameters.

### 📬 Dataset Preparation

Preparing annotations is required to train and test on public dataset or dataset of your own.

- [ ] TODO: Provide dataset preprocessing tool.

Each line in an annotation file should be:

```
[img_path]\t[img_label]
```

For more flexible combination of training data, we use `label_map`. When adding new data classes, you may simply assign the new class with a new label, and use the `label_map` to map these selected classes' labels as desired for training.

For example, if we predefine class name and class labels as:

| Class      | Label |
| ---------- | ----- |
| Real       | 0     |
| ProGAN     | 1     |
| StyleGAN   | 2     |
| SNGAN      | 3     |
| InfoMaxGAN | 4     |

For an experiment we hope to only use Real + ProGAN + SNGAN for training, we may set the `label_map` as follows:

```python
label_map = {
    "0": 0,
    "1": 1,
    "3": 2
}
```

Or if we hope to perform only real or fake classification task, we may set the `label_map` as follows:

```python
label_map = {
    "0": 0,
    "1": 1,
    "2": 1,
    "3": 1
}
```

### 💪 Train

```bash
cd exps/baselines
#             config          work_dir         gpu_id
bash train.sh 01_resnet18.py 01_resnet18_1205 1
```

The checkpoints and logs will be saved in `exps/baselines/01_resnet18_1205`.

### 🧪 Test

```bash
#            config         work_dir         test_tag gpu_id
bash test.sh 01_resnet18.py 01_resnet18_1205 setting1 1
```

The last checkpoint will be automatically loaded for training. Please check `tools/inference.py` if you hope to automatically load the best ckpt for it's temporally hard coded here. If you hope to load a specific ckpt, please add `--ckpt_path=[path]` in `test.sh`.

The inferece results include a .pkl file, a .png confusion matrix figure, a .txt file with evaluation metrics like acc and f1, all named with the provided test_tag in `test.sh` and will be saved in `exp/baselines/01_resnet18_1205/preds`

- [ ] TODO: remove hard code in inference

### 🔎 Vis Frequency

```bash
#                config           work_dir         gpu_id
bash vis_freq.sh vis_frequency.py 01_resnet18_1205 1
```

Note that the `CLASSES` should not be None for the saved samples per class will be named using provided class_names. And the average frequency spectrum per class and samples per class will be saved in `exp/baselines/01_resnet18_1205`.

### 📊 Vis t-SNE

To visualize t-SNE, you need to first implement the filepath and classnum in `tools/create_tsne_list.py` and set the test_meta in your config to be the tsne_list that you just generated.

```bash
#                config         work_dir         test_tag gpu_id
bash vis_tsne.sh 01_resnet18.py 01_resnet18_1205 setting1 1
```

The .pdf and .png format of t-SNE results will be saved in `exps/baselines/01_resnet18_1205/tsne` with name `setting1_xxx`.

## 🗂 Code Structure

<details>
<summary>Click to view the code structure</summary>

```
ModelAttr
├── data  # dataset and dataloader
│   ├── dataset_entry
│   ├── datasets
│   ├── freq_dataset
│   ├── ...
│   └── transforms
│
├── exps  # working directory
│
├─ losses  # used losses
│   ├── autoweight_loss
│   ├── cross_entropy
│   ├── ...
│   └── loss_entry
│
├─ models  # different models
│   ├── classifier
│   ├── attnet  # attnet
│   ├── clip_classifier  # UniFD
│   ├── patch_cnn  # PatchCNN
│   ├── simple_cnn  # LeveFreq
│   ├── ...
│   ├── build_classifiers
│   └── model_entry
│
├── tools  # training relevent
│   ├── evaluation  # evaluation functions
│   ├── inference  # model inference
│   ├── train_cls  # model training
│   ├── inference  # model inference
│   ├── vis_frequency  # visualize frequency spectrum
│   └── vis_tsne  # visualize t-SNE
│
├── utils  # other tools
│   ├── checkpoint  # saving and loading model
│   ├── dist  # distributed training, not fully implemented
│   ├── freq_transform  # FFT / DCT to transform to frequency
│   ├── init_config  # config relevent
│   ├── log  # logger setting
│   ├── metric  # evaluation metrics
│   ├── parser  # parse args from .sh bash file
│   ├── scheduler  # lr scheduler
│   ├── tnse  # t-SNE visualization
│   └── writer  # tensorboard writer
│
├── test_classifier  # test entry
└── train_classifier  # train_entry
```

</details>

<details>
<summary>Click to view the work_dir structure</summary>

```
work_dir
├── checkpoints  # saved ckpts
├── preds # prediction results
│   ├── [tag]_confusion_matrix.png
│   ├── [tag]_metric.txt
│   ├── [tag].pkl
│   └── ...
├── logs
├── tensorlogs
├── tsne  	# if used
├── frequency  	# if used
└── config.py
```

</details>

## ⏳ TODO

- [ ] Public benckmark results.
- [ ] There still exists some hard codes in this codebase.
- [ ] Maybe optimize the code structure to avoid dealing with special cases in `train_cls.py`.
- [ ] Maybe support more methods 🤔

## 📝 Notes

- Some methods is not fully implemented.
  - For [DNA-Det](https://github.com/ICTMCG/DNA-Det) we currently only implemented the second stage of training.
  - For [LGrad](https://github.com/chuangchuangtan/LGrad), it's required to preprocess the image into gradient format. You may use [img2grad](https://github.com/chuangchuangtan/LGrad/blob/master/img2gad_pytorch/gen_imggrad.py) to extract gradient images first. (We tried to implement a model that get gradients during training and testing but it's much slower and contains potential bugs ...)
- Dataset and Results.
  - This codebase is originally developed for evaluating existing methods on a private dataset so the datalist is not currently avaible (We may release later). The results of these models are not fully tested on the public benchmark now. So the `exps/expdata/example.py` is just a placeholder now. You may replace with your own datalist following the example's format. After releasing the used dataset we will implete this

## 🏆 Acknowledgement

This codebase is mainly developed based on [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch), [DNA-Det](https://github.com/ICTMCG/DNA-Det), and repositories from the implemented methods: [Attnet](https://github.com/ningyu1991/GANFingerprints), [LeveFreq](https://github.com/RUB-SysSec/GANDCTAnalysis), [PatchCNN](https://github.com/chail/patch-forensics), [UniFD](https://github.com/Yuheng-Li/UniversalFakeDetect), [LGrad](https://github.com/chuangchuangtan/LGrad).
