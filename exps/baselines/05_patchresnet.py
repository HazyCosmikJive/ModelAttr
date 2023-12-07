'''
PatchCNN - ECCV2020
    [1] L. Chai, D. Bau, S.-N. Lim, and P. Isola, “What Makes Fake Images Detectable? Understanding Properties that Generalize,” in Computer Vision – ECCV 2020, vol. 12371, A. Vedaldi, H. Bischof, T. Brox, and J.-M. Frahm, Eds., in Lecture Notes in Computer Science, vol. 12371. , Cham: Springer International Publishing, 2020, pp. 103–120. doi: 10.1007/978-3-030-58574-7_7.
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
        name = "patch_cnn",
    ),
    encoder = "resnet18",
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