'''
AttNet - ICCV2019
    Yu, Ning, Larry S. Davis, and Mario Fritz. "Attributing fake images to gans: Learning and analyzing gan fingerprints." Proceedings of the IEEE/CVF international conference on computer vision. 2019.
'''

_base_ = [
    '../expdata/example.py',
]

common = dict(
    random_seed = 0,
    use_tensorlog = True,
    task_type = "classifier"
)

model = dict(
    name = "classifer",
    classifier = dict(
        name = "attnet",
    ),
    class_num = 7,
    loss = dict(
        types = "ce"
    )
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