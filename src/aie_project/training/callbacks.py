import optuna
from transformers import TrainerCallback


class DistributedOptunaCallback(TrainerCallback):
    def __init__(self, trial: optuna.Trial, metric_name="eval_loss"):
        self.trial = trial
        self.metric_name = metric_name

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_score = metrics.get(self.metric_name)

        if current_score is None:
            return

        self.trial.report(current_score, step=state.global_step)

        if self.trial.should_prune():
            raise optuna.TrialPruned()
