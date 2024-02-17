import os
import gradio as gr
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
    logger.info("Load model from: {}".format(args.ckpt))
    model = VisionEncoderDecoderModel.from_pretrained(
        args.ckpt
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.ckpt)
    
    
    def inference(input_image):
        # Load image
        logger.info("\nLoad image from: {}".format(input_image))
        image = Image.open(input_image)
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
        logger.info("Output: {}".format(sequence))
        return sequence
    
    
    def clear_inputs():
        return None, None, None
    
    
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown(
                """
                <p align='center' style='font-size: 25px;'>Sumen Latex OCR Model </p>
                """
                
            )
        with gr.Row():
            input_image = gr.Image(source="upload", type="filepath", label="Input Image")
        with gr.Row():
            run_button = gr.Button(value="Run")
            with gr.Column():
                clear_button = gr.Button("Clear")
        with gr.Row():
            output_answer = gr.Textbox(label="Output (latex)")
        ips = input_image
        run_button.click(fn=inference, inputs=ips, outputs=output_answer)
        clear_button.click(fn=clear_inputs, inputs=None, outputs=[input_image, output_answer])
        gr.Examples(
            examples=[
                ["assets/example_1.png"],
                ["assets/example_2.png"],
                ["assets/example_3.png"],
                ["assets/example_4.bmp"],
                ["assets/example_5.bmp"],
                ["assets/example_6.bmp"],
                ["assets/example_7.bmp"],
            ],
            inputs=input_image,
            outputs=None,
            fn=None,
            cache_examples=False,
            examples_per_page=5,
        )
    
    block.launch(debug=True, share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sumen Latex OCR")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/sumen-small-checkpoint-7450",
        help="Path to the checkpoint. (.npz)",
    )
    args = parser.parse_args()
    main(args)