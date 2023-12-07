'''
DNADet - AAAI2022
    Yang, Tianyun, et al. "Deepfake network architecture attribution." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 4. 2022.
'''

_base_ = [
    '../expdata/example.py',        # you need to implement the example.py first
]

common = dict(
    random_seed = 0,
    use_tensorlog = True,
    task_type = "classifier"
)

data = dict(
    dataset_type = "dnadet_dataset",
    resize_size = (512, 512),
    crop_size = (128, 128),
    second_resize_size = None,
    multi_size = [(64, 64)] * 16,
    crop_samples = 16,
)

model = dict(
    name = "classifer",
    classifier = dict(
        name = "supcontrast_simplecnn",
    ),
    class_num = 7,
    loss = dict(
        types = ["ce", "supcontrast"],
        weights = [1.0, 0.1],
        temperature = 0.07,
        use_crops = True,
    ),
    # awl = True,
)

trainer = dict(
    optimizer = dict(
        type = "AdamW",
        lr = 0.0005,
        momentum = 0.9,
        weight_decay = 0.0001,
        nesterov = True,
    ),
    lr_scheduler = dict(
        type = "cosine",
        warmup_epoch = 1,
    ),
    epoch = 50,
    batchsize = 32,
    workers = 0,
    print_freq = 50,
    save_freq = 5,
    metric = "acc",
    early_stop = True,
    early_stop_bar = 20,
)