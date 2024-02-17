import nltk
import evaluate
from nltk import edit_distance as compute_edit_distance
from src.utils.common_utils import compute_exprate


class Metrics:
    def __init__(self, processor):
        self.processor = processor
        self.bleu = evaluate.load("bleu")
        self.exact_match = evaluate.load("exact_match")
        self.wer = evaluate.load("wer")
        
    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        
        total_edit_distance, total_bleu, total_exact_match = 0, 0, 0
        for i in range(len(pred_str)):
            # Compute edit distance score
            edit_distance = compute_edit_distance(
                pred_str[i], 
                label_str[i]
            )/max(len(pred_str[i]),len(label_str[i]))
            total_edit_distance += edit_distance
            # Compute bleu score
            try:
                bleu = self.bleu.compute(
                    predictions=[pred_str[i]],
                    references=[label_str[i]],
                    max_order=4 # Maximum n-gram order to use when computing BLEU score
                )
                total_bleu += bleu['bleu']
            except ZeroDivisionError:
                total_bleu += 0
            # Compute exact match score
            exact_match = self.exact_match.compute(
                predictions=[pred_str[i]],
                references=[label_str[i]],
                # regexes_to_ignore=[' '],
            )
            total_exact_match += exact_match['exact_match']
        bleu = total_bleu / len(pred_str)
        exact_match = total_exact_match / len(pred_str)
        # Convert minimun edit distance score to maximun edit distance score
        edit_distance = 1 - (total_edit_distance / len(pred_str))
        # Compute word error rate score
        wer = self.wer.compute(predictions=pred_str, references=label_str)
        # Compute expression rate score
        exprate = compute_exprate(predictions=pred_str, references=label_str)
        
        return {
            "bleu": round(bleu*100, 2),
            "maximun_edit_distance": round(edit_distance*100, 2),
            "exact_match": round(exact_match*100, 2),
            "wer": round(wer*100, 2),
            "exprate": round(exprate*100, 2),
        }