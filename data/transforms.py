import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms


# TODO: aug paramsfrom config
def get_train_transforms(config):
    # trans_list = [
    #     A.Resize(config.data.image_size, config.data.image_size, p=1.0),
    # ]
    trans_list = []
    if config.data.transform.get("color", False):
        trans_list.append(
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
            ], p=0.5),
        )
    if config.data.transform.get("blur", False):
        blur_limit = config.data.transform.get('blur_limit', [1, 5])
        trans_list.append(
            A.OneOf([
                A.Blur(blur_limit=blur_limit, p=0.5),
                A.GaussianBlur(p=0.5),
            ], p=0.5)
        )
    if config.data.transform.get("noise", False):
        noise_var_limit = config.data.transform.get('noise_var_limit', (5, 15))
        trans_list.append(
            A.GaussNoise(var_limit=noise_var_limit, p=0.5)
        )
    if config.data.transform.get("jpeg", False):
        jpeg_quality = config.data.transform.get('jpeg_quality', (80, 100))
        trans_list.append(
            A.JpegCompression(quality_lower=jpeg_quality[0], quality_upper=jpeg_quality[1], p=0.5)
        )
    
    return A.Compose(trans_list)

# only normalize in validation
def get_val_transforms(config):
    return A.Compose([
        # A.Resize(config.data.image_size, config.data.image_size, p=1.0),
        A.Resize(config.data.transform.resize_size[0], config.data.transform.resize_size[1], p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], p=1.0),
        ToTensorV2(transpose_mask=True),
    ], p=1.)