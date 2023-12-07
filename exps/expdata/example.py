data = dict(
    # support dataset_type:
    #     image_dataset: spatial input
    #     freq_dataset: frequency input, use DCT or FFT freq transform
    #     dnadet_dataset: for DNA-Det only, return cropped images
    dataset_type = "image_dataset",
    # label_map is a dict to map current data label to desired data labels.
    # this design is for flexible combination of different training / testing datas
    label_map={
        "0": 0,
        "1": 1,
        "6": 1,
        "7": 1,
        "9": 2,
        "10": 2,
        "14": 2,
        "2": 3,
        "28": 3,
        "29": 3,
        "17": 4,
        "30": 4,
        "31": 4,
        "16": 5,
        "32": 5,
        "33": 5,
        "27": 6,
        "35": 6,
        "36": 6,
    },
    # datalist here.
    # each line is: "imgpath \t imglabel"
    train_meta = [
        '/data/datalist/coco_train.txt', # coco, 0 -> 0
        'TO BE REPLACES'
    ],
    val_meta = [
        '/data/datalist/coco_val.txt', # coco, 0 -> 0
        'TO BE REPLACES'
    ],
    test_meta = [
        '/data/datalist/coco_test.txt', # coco, 0 -> 0
        'TO BE REPLACES'
    ],
    transform = dict(
        first_crop_size = (256, 256),  #  1. crop first
        resize_size = (512, 512),  # 2. then resize
    ),
    CLASSES = None # or, ["real", "ProGAN", "StyleGAN", ...]
)