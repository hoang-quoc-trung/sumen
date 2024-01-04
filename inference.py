import torch
import logging
import argparse
from PIL import Image
from transformers import AutoProcessor
from transformers import VisionEncoderDecoderModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args):
    # Get the device
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        logger.info("There are {} GPU(s) available.".format(torch.cuda.device_count()))
        logger.info('We will use the GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        logger.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Init model
    model = VisionEncoderDecoderModel.from_pretrained(
        args.ckpt
    ).to(device)

    # Init processor
    processor = AutoProcessor.from_pretrained(args.ckpt)

    # Load image
    image = Image.open(args.input_image)
    if not image.mode == "RGB":
        image = image.convert('RGB')

    pixel_values = processor.image_processor(
        image,
        return_tensors="pt",
        data_format="channels_first",
    ).pixel_values
    task_prompt = processor.tokenizer.bos_token
    decoder_input_ids = processor.tokenizer(
        task_prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids
    
    # Generate LaTeX expression
    with torch.no_grad():
        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_length,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=4,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    sequence = processor.tokenizer.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(
            processor.tokenizer.eos_token, ""
        ).replace(
            processor.tokenizer.pad_token, ""
        ).replace(processor.tokenizer.bos_token,"")
    logger.info("\nOutput: {}".format(sequence))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--input_image",
        type=str,
        default="assets/example_1.png",
        help="Path to image file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="src/checkpoints",
        help="Path to checkpoint model",
    )
    args = parser.parse_args()
    main(args)