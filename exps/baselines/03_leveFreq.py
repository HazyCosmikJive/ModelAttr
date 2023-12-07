'''
LeveFreq - ICML2020
    Frank, Joel, et al. "Leveraging frequency analysis for deep fake image recognition." International conference on machine learning. PMLR, 2020.
'''

_base_ = [
    '../expdata/example.py',        # you need to implement the example.py first
]

common = dict(
    random_seed = 0,
    use_tensorlog = True,
    task_type = "classifier"
)

model = dict(
    name = "classifer",
    classifier = dict(
        name = "simplecnn",
    ),
    class_num = 7,
    loss = dict(
        types = "ce"
    )
)

data = dict(
    dataset_type = "freq_dataset",
    freq_transform = "DCT",
    gray = False,
    logscale_factor=20,
)

trainer = dict(
    optimizer = dict(
        type = "AdamW",
        lr = 0.0005,
    ),
    lr_scheduler = dict(
        type = "cosine",
        warmup_epoch = 1,
    ),
    epoch = 50,
    batchsize = 32,
    workers = 8,
    print_freq = 50,
    save_freq = 5,
    metric = "acc",
    early_stop = True,
    early_stop_bar = 20,
)