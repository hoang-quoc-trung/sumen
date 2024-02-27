import os
import csv
import torch
import numpy


def check_device(logger=None):
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        logger.info("There are {} GPU(s) available.".format(torch.cuda.device_count()))
        logger.info('We will use the GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        logger.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def print_trainable_parameters(model, logger):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        "Total params: {}M ({}) || Trainable params: {} || Trainable: {}%".format(
            round(all_param/1000000),
            all_param,
            trainable_params,
            100 * trainable_params / all_param
        )
    )


def save_log(
    loss: float,
    bleu: float,
    edit_distance: float,
    exact_match: float,
    wer: float,
    exprate: float,
    exprate_error_1: float,
    exprate_error_2: float,
    exprate_error_3: float,
    file_name="test_log.csv",
):
    
    os.makedirs('log', exist_ok=True)
    file_path = os.path.join('log', file_name)
    with open(file_path, mode="a", newline="") as csv_file:
        fieldnames = [
            "loss",
            "bleu",
            "edit_distance",
            "exact_match",
            "wer",
            "exprate",
            "exprate_error_1",
            "exprate_error_2",
            "exprate_error_3"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        # write the header row
        if csv_file.tell() == 0:
            writer.writeheader()
        # write the data row
        writer.writerow(
            {
                "loss": loss,
                "bleu": bleu,
                "edit_distance": edit_distance,
                "exact_match": exact_match,
                "wer": wer,
                "exprate": exprate,
                "exprate_error_1": exprate_error_1,
                "exprate_error_2": exprate_error_2,
                "exprate_error_3": exprate_error_3,
            }
        )


def cmp_result(label,rec):
    dist_mat = numpy.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)
    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)


def compute_exprate(predictions, references):
    total_label = 0
    total_line = 0
    total_line_rec = 0
    total_line_error_1 = 0
    total_line_error_2 = 0 
    total_line_error_3 = 0
    for i in range(len(references)):
        pre = predictions[i].split()
        ref = references[i].split()
        dist, llen = cmp_result(pre, ref)
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec += 1
        elif dist ==1:
            total_line_error_1 +=1
        elif dist ==2:
            total_line_error_2 +=1
        elif dist ==3:
            total_line_error_3 +=1
    exprate = float(total_line_rec)/total_line
    error_1 = float(
        total_line_error_1 + total_line_rec
    )/total_line
    error_2 = float(
        total_line_error_2 + total_line_error_1 +total_line_rec
    )/total_line
    error_3 = float(
        total_line_error_3 + total_line_error_2 + total_line_error_1 + total_line_rec
    )/total_line
    return exprate, error_1, error_2, error_3