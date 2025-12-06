import evaluate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


metric_acc = evaluate.load("accuracy")
metric_precision = evaluate.load("precision")
metric_recall = evaluate.load("recall")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    # for multi task (need to unpack)
    predictions, labels = eval_pred
    mat_logits, trans_logits = predictions
    mat_labels, trans_labels = labels

    # helper function for code reuse
    def calculate_single_head(logits, ground_truth, prefix):
        preds = np.argmax(logits, axis=-1)

        acc = metric_acc.compute(predictions=preds, references=ground_truth)
        prec = metric_precision.compute(predictions=preds, references=ground_truth, average='weighted')
        rec = metric_recall.compute(predictions=preds, references=ground_truth, average='weighted')
        f1_w = metric_f1.compute(predictions=preds, references=ground_truth, average='weighted')
        f1_m = metric_f1.compute(predictions=preds, references=ground_truth, average='macro')

        return {
            f'{prefix}_accuracy': acc['accuracy'],
            f'{prefix}_precision': prec['precision'],
            f'{prefix}_recall': rec['recall'],
            f'{prefix}_f1_weighted': f1_w['f1'],
            f'{prefix}_f1_macro': f1_m['f1'],
        }

    mat_results = calculate_single_head(mat_logits, mat_labels, "mat")
    trans_results = calculate_single_head(trans_logits, trans_labels, "trans")
    return {**mat_results, **trans_results, "combined_f1_macro": (mat_results['mat_f1_macro'] + trans_results['trans_f1_macro']) / 2}
