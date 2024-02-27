import os
import yaml
import logging
import argparse
import pandas as pd
import huggingface_hub
from src.utils import common_utils
from src.utils.metrics import Metrics
from src.dataset.data_loader import Sumen_Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    VisionEncoderDecoderModel,
    AutoProcessor
)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s\t%(levelname)s\t%(name)s %(filename)s:%(lineno)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["WANDB_DISABLED"] = "true"


def main(args):
    assert os.path.exists(
        args.config_path
    ), f"Config file {args.config_path} does not exist!"
    # Load the config file
    config = yaml.safe_load(open(args.config_path))
    
    # Login to huggingface hub
    huggingface_hub.login(config['huggingface']['hub_token'])

    # Get the device
    device = common_utils.check_device(logger)
    
    # Init model
    model = VisionEncoderDecoderModel.from_pretrained(
        args.ckpt
    ).to(device)

    # Init processor
    processor = AutoProcessor.from_pretrained(args.ckpt)
    
    # Load dataset
    test_df = pd.read_csv(config['datasets']['test']['dataframe_path'])
    logger.info("Total test dataset: {}".format(len(test_df)))
    
    test_dataset = Sumen_Dataset(
        test_df,
        phase='test',
        root_dir=config['datasets']['test']['images_root'],
        tokenizer=processor.tokenizer,
        processor=processor.image_processor,
        max_length=config['hyperparams']['max_length'],
        image_size=config['hyperparams']['image_size'],
    )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['hyperparams']['save_dir'],
        seed=config['hyperparams']['random_seed'],
        data_seed=config['hyperparams']['random_seed'],
        per_device_eval_batch_size=config['datasets']['eval']['batch_size'],
        predict_with_generate=True, 
        do_train=False,
        do_eval=True,
        report_to=None,
        dataloader_num_workers=config['datasets']['num_workers'],
        eval_accumulation_steps=config['hyperparams']['gradient_accumulation_steps'],
    )
    
    metrics = Metrics(processor)
    
    trainer = Seq2SeqTrainer(
        tokenizer=processor,
        model=model,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        eval_dataset=test_dataset,
        data_collator=default_data_collator,
    )
    
    metrics = trainer.evaluate()
    common_utils.save_log(
        loss=metrics['eval_loss'],
        bleu=metrics['eval_bleu'],
        edit_distance=metrics['eval_maximun_edit_distance'],
        exact_match=metrics['eval_exact_match'],
        wer=metrics['eval_wer'],
        exprate=metrics['eval_exprate'],
        exprate_error_1=metrics['eval_exprate_error_1'],
        exprate_error_2=metrics['eval_exprate_error_2'],
        exprate_error_3=metrics['eval_exprate_error_3'],
    )
    logger.info("Results: {}".format(metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument(
        '--config_path',
        default='src/config/base_config.yaml',
        help="Path to the config file",
        type=str,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="src/checkpoints",
        help="Path to checkpoint model",
    )
    args = parser.parse_args()
    main(args)