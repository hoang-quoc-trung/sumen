import os
import torch
import logging
import argparse
import streamlit as st
import nltk
import evaluate
from PIL import Image
from transformers import AutoProcessor
from transformers import VisionEncoderDecoderModel
from src.utils import common_utils
from nltk import edit_distance as compute_edit_distance
from src.utils.common_utils import compute_exprate

bleu_func = evaluate.load("bleu")
wer_func = evaluate.load("wer")
exact_match_func = evaluate.load("exact_match")
        
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args):
    @st.cache_resource
    def init_model():
        # Get the device
        device = common_utils.check_device(logger)
        # Init model
        logger.info("Load model & processor from: {}".format(args.ckpt))
        model = VisionEncoderDecoderModel.from_pretrained(
            args.ckpt
        ).to(device)
        # Load processor
        processor = AutoProcessor.from_pretrained(args.ckpt)
        task_prompt = processor.tokenizer.bos_token
        decoder_input_ids = processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids
        return model, processor, decoder_input_ids, device
    
    model, processor, decoder_input_ids, device = init_model()
    
    @st.cache_data
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
        # Generate LaTeX expression
        with torch.no_grad():
            outputs = model.generate(
                pixel_values.to(device),
                decoder_input_ids=decoder_input_ids.to(device),
                max_length=model.decoder.config.max_length,
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
    
    @st.cache_data
    def compute_crohme_metrics(label_str, pred_str):
        wer = wer_func.compute(predictions=[pred_str], references=[label_str])
        # Compute expression rate score
        exprate, error_1, error_2, error_3 = compute_exprate(
            predictions=pred_str,
            references=label_str
        )
        return round(wer*100, 2), round(exprate*100, 2), round(error_1*100, 2), round(error_2*100, 2), round(error_3*100, 2)
    
    
    @st.cache_data
    def compute_img2latex100k_metrics(label_str, pred_str):
        # Compute edit distance score
        edit_distance = compute_edit_distance(
            pred_str, 
            label_str
        )/max(len(pred_str),len(label_str))
        # Convert minimun edit distance score to maximun edit distance score
        edit_distance = round((1 - edit_distance)*100, 2)
        # Compute bleu score
        bleu = bleu_func.compute(
            predictions=[pred_str],
            references=[label_str],
            max_order=4 # Maximum n-gram order to use when computing BLEU score
        )
        bleu = round(bleu['bleu']*100, 2)
        exact_match = exact_match_func.compute(
            predictions=[pred_str],
            references=[label_str]
        )
        exact_match = round(exact_match['exact_match']*100, 2)
        return bleu, edit_distance, exact_match

    # --------------------------------- Sreamlit code ---------------------------------

    st.markdown("<h1 style='text-align: center; color: LightSkyBlue;'>Math Formula Images To LaTeX Code Based On End-to-End Approach With Attention Mechanism</h1>", unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.write('')
    st.header('Input', divider='blue')
    uploaded_file = st.file_uploader(
            "Upload an image",
            type = ['png', 'jpg'],
        )
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        st.image(
            bytes_data,
            width = 700,
            channels = 'RGB',
            output_format = 'PNG'
        )
    on = st.toggle('Enable testing with label')

    if on:
        with st.container(border=True):
            option = st.selectbox(
                'Benchmark ?',
                ('Im2latex-100k', 'CROHME'))
            label = st.text_input('Label', None)
        run = st.button("Run")
        
        if run is True and uploaded_file is not None and label is not None and option == 'Im2latex-100k':
            pred_str = inference(uploaded_file)
            st.header('Output', divider='blue')
            st.latex(pred_str)
            st.write(':orange[Latex sequences:]', pred_str)
            bleu, edit_distance, exact_match = compute_img2latex100k_metrics(label, pred_str)
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Bleu", bleu)
                col2.metric("Edit Distance", edit_distance)
                col3.metric("Exact Match", exact_match)
            
        if run is True and uploaded_file is not None and label is not None and option == 'CROHME':
            pred_str = inference(uploaded_file)
            st.header('Output', divider='blue')
            st.latex(pred_str)
            st.write(':orange[Latex sequences:]', pred_str)
            wer, exprate, error_1, error_2, error_3 = compute_crohme_metrics(label, pred_str)
            with st.container(border=True):
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("ExpRate", exprate)
                col2.metric("ExpRate 1", error_1)
                col3.metric("ExpRate 2", error_2)
                col4.metric("ExpRate 3", error_3)
                col5.metric("WER", wer)

    else:
        run = st.button("Run")
        if run is True and uploaded_file is not None:
            pred_str = inference(uploaded_file)
            st.write('')
            st.header('Output', divider='blue')
            st.latex(pred_str)
            st.write(':orange[Latex sequences:]', pred_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sumen Latex OCR")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/sumen-base",
        help="Path to the checkpoint",
    )
    args = parser.parse_args()
    main(args)