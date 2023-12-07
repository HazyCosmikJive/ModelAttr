from models.build_classifiers import build_classifier


def model_entry(config, logger):
    model = None

    task_type = config.common.task_type

    if task_type == "classifier":
        model = build_classifier(config)

        # fix encoder
        if config.model.get("fix_encoder", False):
            for n, p in model.named_parameters():
                if "encoder" in n:
                    p.requires_grad = False
            logger.info("[MODEL] Fix encoder.")

        if len(config.model.get("fix_encoder_partial", [])) > 0:
            for n, p in model.named_parameters():
                for layer_idx in config.model.get("fix_encoder_partial", []):
                    if "encoder.layer{}".format(layer_idx) in n:
                        p.requires_grad = False
            logger.info("[MODEL] Fix encoder layer {}.".format(config.model.get("fix_encoder_partial", [])))

        logger.info("[MODEL] Build Model Done.")
        return model

    else:
        raise NotImplementedError("Model for task type {} is not implemented now.".format(task_type))


