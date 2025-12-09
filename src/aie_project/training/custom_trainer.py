import numpy as np
import wandb
from transformers import Trainer


class VisualizationTrainer(Trainer):
    # Custom trainer to log wandb confusion matrix when evaluating
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # standard evaluate function with added confusion matrix logging
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        output = self.predict(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        if self.is_world_process_zero() and "wandb" in self.args.report_to:
            # log confusion matrix if main process and wandb is enabled
            self.log_confusion_matrix(output.predictions, output.label_ids)

        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        return output.metrics

    def log_confusion_matrix(self, predictions, label_ids):
        # unpack predictions and labels if necessary
        if isinstance(predictions, tuple):
            logits_mat, logits_trans = predictions
            labels_mat, labels_trans = label_ids
        else:
            logits_mat = predictions
            labels_mat = label_ids
            logits_trans = None

        preds_mat = np.argmax(logits_mat, axis=-1)
        # try int and string keys because JSON object keys are always strings
        try:
            mat_names = [self.model.config.id2label_1[i] for i in range(len(self.model.config.id2label_1))]
        except KeyError:
            mat_names = [str(i) for i in range(len(self.model.config.id2label_1))]
        wandb.log({
            "confusion_matrix/material": wandb.plot.confusion_matrix(
                probs=None, y_true=labels_mat, preds=preds_mat, class_names=mat_names
            )
        })

        if logits_trans is not None:
            preds_trans = np.argmax(logits_trans, axis=-1)
            try:
                trans_names = [self.model.config.id2label_2[i] for i in range(len(self.model.config.id2label_2))]
            except KeyError:
                trans_names = [str(i) for i in range(len(self.model.config.id2label_2))]
            wandb.log({
                "confusion_matrix/transparency": wandb.plot.confusion_matrix(
                    probs=None, y_true=labels_trans, preds=preds_trans, class_names=trans_names
                )
            })
