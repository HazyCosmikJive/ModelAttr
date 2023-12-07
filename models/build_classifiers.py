# TODO: better ways for selecting model
from models.classifiers import *
from models.attnet import AttNet_formal
from models.lgrad import LGrad
import models.patch_cnn as patch_cnn
from models.clip_classifier import *
from models.simple_cnn import Simple_CNN, SupConNet


def build_classifier(config):
    classifier_name = config.model.classifier.name.lower()
    if "basic_classifier" == classifier_name:
        classifier = BasicClassifier(config)
    elif "mlp_classifier" == classifier_name:
        classifier = MLPClassifier(config)
    #! baseline here
    elif "unifd" == classifier_name:
        classifier = CLIPLinear(config)
    elif "attnet" == classifier_name:
        classifier = AttNet_formal(config)
    elif "lgrad" == classifier_name:
        classifier = LGrad(config)
    elif "patch_cnn" == classifier_name:
        # TODO: remove hard code
        if config.model.encoder == "resnet18":
            classifier = patch_cnn.make_patch_resnet(depth=18, layername='layer1', num_classes=config.model.class_num)
        else:
            raise NotImplementedError("Temporally hard coded for resnet18 only.")
    elif "simplecnn" == classifier_name:
        classifier = Simple_CNN(config)
    elif "supcontrast_simplecnn" == classifier_name:
        backbone = Simple_CNN(config)
        classifier = SupConNet(backbone)
    else:
        raise NotImplementedError("Classifier {} is not implemented now.".format(classifier_name))
    return classifier