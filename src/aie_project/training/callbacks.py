import optuna
import wandb
from transformers import TrainerCallback


class DistributedOptunaCallback(TrainerCallback):
    def __init__(self, trial: optuna.Trial, objective_metric: str = "eval_combined_f1_macro"):
        self.trial = trial
        self.objective_metric = objective_metric

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.trial.set_user_attr(key, value)

        self.trial.report(metrics[self.objective_metric], step=int(state.epoch))
        if self.trial.should_prune():
            if args.local_rank in [-1, 0]:
                wandb.log({"trial_pruned": True})
                wandb.finish()
            raise optuna.TrialPruned()
