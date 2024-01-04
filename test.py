import os
import yaml
import logging
import argparse
import pandas as pd
from src.models import sumen_model
from src.utils.metrics import Metrics
from src.dataset.data_loader import Nougat_Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
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
    
    config['hyperparams']['pretrained_model_name_or_path'] = args.ckpt
    
    model, processor = sumen_model.init_model(config, logger)
    
    test_df = pd.read_csv(config['datasets']['test']['dataframe_path'])
    logger.info("Total test dataset: {}".format(len(test_df)))
    
    test_dataset = Nougat_Dataset(
        test_df,
        phase='test',
        root_dir=config['datasets']['test']['images_root'],
        tokenizer=processor.tokenizer,
        processor=processor.image_processor,
        max_length=config['hyperparams']['max_length'],
    )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['hyperparams']['save_dir'],
        seed=config['hyperparams']['random_seed'],
        data_seed=config['hyperparams']['random_seed'],
        per_device_eval_batch_size=config['datasets']['eval']['batch_size'],
        predict_with_generate=True, 
        do_train=False,
        do_eval=True,
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