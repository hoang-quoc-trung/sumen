import os
import yaml
import wandb
import torch
import logging
import argparse
import transformers
import pandas as pd
from lion_pytorch import Lion
from src.models import sumen_model
from src.utils.metrics import Metrics
from src.dataset.data_loader import Sumen_Dataset
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

os.environ["WANDB_DISABLED"] = "false"


def main(args):
    # Load the config file
    assert os.path.exists(
        args.config_path
    ), f"Config file {args.config_path} does not exist!"
    config = yaml.safe_load(open(args.config_path))
    
    # Initialize wandb
    wandb.login(key=config['wandb']['key'])
    wandb.init(project=config['wandb']['project_name'])
    
    # Initialize model
    model, processor = sumen_model.init_model(config, logger)
    
    # Load dataset for training
    train_df = pd.read_csv(config['datasets']['train']['dataframe_path'])
    logger.info("Total train dataset: {}".format(len(train_df)))
    val_df = pd.read_csv(config['datasets']['eval']['dataframe_path'])
    logger.info("Total eval dataset: {}".format(len(val_df)))
    
    # Initialize dataloader
    train_dataset = Sumen_Dataset(
        train_df,
        phase='train',
        root_dir=config['datasets']['train']['images_root'],
        tokenizer=processor.tokenizer,
        processor=processor.image_processor,
        max_length=config['hyperparams']['max_length'],
        image_size=config['hyperparams']['image_size'],
    )
    val_dataset = Sumen_Dataset(
        val_df,
        phase='eval',
        root_dir=config['datasets']['eval']['images_root'],
        tokenizer=processor.tokenizer,
        processor=processor.image_processor,
        max_length=config['hyperparams']['max_length'],
        image_size=config['hyperparams']['image_size'],
    )
    
    # Config trainer 
    # Detail: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['hyperparams']['save_dir'],
        overwrite_output_dir=True,
        seed=config['hyperparams']['random_seed'],
        data_seed=config['hyperparams']['random_seed'],
        per_device_train_batch_size=config['datasets']['train']['batch_size'],
        per_device_eval_batch_size=config['datasets']['eval']['batch_size'],
        predict_with_generate=True,
        evaluation_strategy="steps",
        eval_steps=config['hyperparams']['eval_steps'],
        save_strategy="steps",
        save_steps=config['hyperparams']['save_steps'], 
        do_train=True,
        do_eval=True,
        logging_first_step=True,
        logging_strategy="steps",
        logging_steps=config['hyperparams']['logging_steps'],  
        num_train_epochs=config['hyperparams']['epochs'],
        save_total_limit=1,
        report_to=['wandb'],
        dataloader_num_workers=config['datasets']['num_workers'],
        gradient_accumulation_steps=config['hyperparams']['gradient_accumulation_steps'],
        eval_accumulation_steps=config['hyperparams']['gradient_accumulation_steps'],
        push_to_hub=True,
        hub_model_id=config['huggingface']['hub_model_id'],
        hub_token=config['huggingface']['hub_token'],
        hub_private_repo=True,
        hub_strategy="checkpoint",
    )
    
    # Initialize Lion optimizer
    optimizer = Lion(
        model.parameters(),
        lr=float(config['hyperparams']['optimizer']['lr']),
        weight_decay=float(config['hyperparams']['optimizer']['weight_decay']),
        betas=(config['hyperparams']['optimizer']['beta1'],
            config['hyperparams']['optimizer']['beta2']
        ),
    )
    
    # Initialize learning rate scheduler
    num_training_steps = (
        config['hyperparams']['epochs'] * len(train_dataset)
    ) // (config['datasets']['train']['batch_size'] * config['hyperparams']['gradient_accumulation_steps'])
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config['hyperparams']['warmup_steps'],
        num_training_steps=num_training_steps,
    )
    
    # Load metrics
    metrics = Metrics(processor)
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        tokenizer=processor,
        model=model,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        optimizers=(optimizer, lr_scheduler),
    )
    
    # Start training
    logger.info("Resume from checkpoint: {}".format(args.resume_from_checkpoint))
    if args.resume_from_checkpoint == 'true':
        trainer.train(resume_from_checkpoint=True) 
    else:
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        '--config_path',
        default='src/config/base_config.yaml',
        help="Path to the config file",
        type=str,
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        default='false',
        help="Continue training from saved checkpoint (true/false)",
        type=str,
    )
    args = parser.parse_args()
    main(args)