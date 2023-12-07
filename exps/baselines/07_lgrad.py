'''
LGrad - CVPR2023
    Tan, Chuangchuang, et al. "Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
'''

_base_ = [
    '../expdata/setting1.py', # data1
]

common = dict(
    random_seed = 0,
    use_tensorlog = True,
    task_type = "classifier"
)

model = dict(
    name = "classifer",
    classifier = dict(
        name = "mlp_classifier",
    ),
    encoder = "resnet50",
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
    batchsize = 16,
    workers = 8,
    print_freq = 50,
    save_freq = 5,
    metric = "acc",
    early_stop = True,
    early_stop_bar = 20,
)