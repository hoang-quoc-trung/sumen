from transformers import (
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    AutoProcessor
)
from peft import inject_adapter_in_model, LoraConfig
from src.utils import common_utils


def init_model(config, logger):
    model_args = config["hyperparams"]
    pretrained_model_name_or_path = model_args["pretrained_model_name_or_path"]
    logger.info("Init weight from: {}".format(pretrained_model_name_or_path))
    # Initialize processor & image processor & tokenizer
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)
    processor.image_processor.size = {
        'height': model_args['image_size'][0],
        'width': model_args['image_size'][1]
    }
    # Model initialization
    config = VisionEncoderDecoderConfig.from_pretrained(pretrained_model_name_or_path)
    config.encoder.image_size = model_args['image_size']
    config.decoder.max_length = model_args['max_length']
    model = VisionEncoderDecoderModel.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        config=config
    )
    model.config.early_stopping = model_args['early_stopping']
    model.config.length_penalty = model_args['length_penalty']
    model.config.num_beams = model_args['num_beams']
    model.config.top_p = model_args['top_p']
    model.config.top_k = model_args['top_k']
    # model.config.no_repeat_ngram_size = 3
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    if model_args['fine_tune_lora_adapter']['enable'] is True:
        lora_config = LoraConfig(
            lora_alpha=model_args['fine_tune_lora_adapter']['alpha'],
            lora_dropout=model_args['fine_tune_lora_adapter']['dropout'],
            r=model_args['fine_tune_lora_adapter']['r'],
            bias="none",
            target_modules=model_args['fine_tune_lora_adapter']['target_modules'],
        )   
        model = inject_adapter_in_model(lora_config, model)
        model.add_adapter(
            lora_config,
            adapter_name=model_args['fine_tune_lora_adapter']['adapter_name']
        )
        model.enable_adapters()
        common_utils.print_trainable_parameters(model, logger)
    else:
        device = common_utils.check_device(logger)
        model = model.to(device)
        total_parameter = sum([param.nelement() for param in model.parameters()])
        logger.info("Number of parameter: {}M ({})".format(round(total_parameter/1000000), total_parameter))
    
    return model, processor