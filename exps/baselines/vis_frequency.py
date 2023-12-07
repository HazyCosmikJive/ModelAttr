_base_ = [
    # '../expdata/example.py',
    '../expdata/setting1.py',
]
common = dict(
    random_seed = 0,
    task_type = "vis_freq"
)

# vis frequency
vis_freq = dict(
    freq_transform_method = "FFT",
    vis_per_class = 100,
    scale_factor = 10,
)


trainer = dict(
    optimizer = dict(
        type = "AdamW",
        lr = 0.0005,
    ),
    lr_scheduler = dict(
        type = "step",
        warmup_epoch = 0,
    ),
    epoch = 50,
    batchsize = 64,
    workers = 0,
    print_freq = 50,
    save_freq = 5,
    metric = "acc",
    early_stop = True,
    early_stop_bar = 10,
)