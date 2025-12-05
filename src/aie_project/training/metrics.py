import evaluate


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    accuracy_result = accuracy.compute(predictions=preds, references=labels)
    precision_result = precision.compute(predictions=preds, references=labels, average='weighted')
    recall_result = recall.compute(predictions=preds, references=labels, average='weighted')
    f1_result = f1.compute(predictions=preds, references=labels, average='weighted')

    return {
        'accuracy': accuracy_result['accuracy'],
        'precision': precision_result['precision'],
        'recall': recall_result['recall'],
        'f1': f1_result['f1'],
    }
