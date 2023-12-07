'''
baseline - resnet18
'''
# import data relevant settings used for training and testing, 
_base_ = [
    '../expdata/example.py',        # you need to implement the example.py first
]

common = dict(
    random_seed = 0,                # random seed for reproducable results
    use_tensorlog = True,
    task_type = "classifier", 
)

# If hope to modify some dataset settings:
# data = dict(
#     test_meta = ["new_test_data.txt"],
# )

model = dict(
    name = "classifer",
    classifier = dict(
        name = "mlp_classifier",    # to select train model, please refer to `models/build_classifiers.py` for all supported models
    ),
    encoder = "resnet18",           # for models implemented with smp, supported encoders can be found in: https://smp.readthedocs.io/en/latest/encoders.html
    class_num = 7,                  # class numbers
    loss = dict(
        types = "ce"                # loss can be a str or a list.
        # an example for multi-loss:
        # types = ["ce", "mse"],
        # weights = [1.0, 1.0],     # you may set loss weights here; weights = [1.0] * len(types) by default.
        # awl = True,               # or use the auto-weighted loss;
    )
)

trainer = dict(
    optimizer = dict(
        type = "AdamW",             # only support "AdamW" now (for convenient)
        lr = 0.0005,
    ),
    lr_scheduler = dict(
        type = "cosine",            # support "cosine" and "step" now.
        warmup_epoch = 1,
    ),
    epoch = 50,
    batchsize = 64,
    workers = 8,
    print_freq = 50,                # freq of printing logs
    save_freq = 5,                  # freq of saving ckpts
    metric = "acc",                 # metric for selecting best model, support "acc", "f1", "recall"
    early_stop = True,              # early stop
    early_stop_bar = 20,
)