'''
UniFD - CVPR2023:
    Ojha, Utkarsh, Yuheng Li, and Yong Jae Lee. "Towards universal fake image detectors that generalize across generative models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
'''

_base_ = [
    '../expdata/setting1.py', # data1
]

common = dict(
    random_seed = 0,
    use_tensorlog = True,
    task_type = "classifier"
)

# for CLIP, input should be 224 * 224
data = dict(
    transform = dict(
        first_crop_size = (224, 224),   # 1. crop first
        resize_size = None,             # 2. no need to resize
    ),
)

model = dict(
    name = "classifer",
    classifier = dict(
        name = "unifd",
    ),
    encoder = "clip",
    clip_loadpath = "openai/clip-vit-base-patch32",  # or replace with your local huggingface CLIP path
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