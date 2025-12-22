import os
import time
import csv
import json


class ExperimentLogger:
    def __init__(self, log_dir: str, run_name: str | None, use_wandb: bool, config: dict | None, wandb_project: str | None = None):
        self.log_dir = log_dir or "runs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_name = run_name or str(int(time.time()))
        self.run_dir = os.path.join(self.log_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_csv_path = os.path.join(self.run_dir, "metrics.csv")
        self._csv_file = open(self.metrics_csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "step",
                "wall_time_sec",
                "train_loss",
                "lr",
                "tokens_per_sec",
                "val_loss",
            ],
        )
        self._csv_writer.writeheader()
        self.run_start_time = time.time()
        self.use_wandb = use_wandb
        self._wandb = None
        if self.use_wandb:
            import wandb
            self._wandb = wandb
            self._wandb_run = self._wandb.init(project=wandb_project or "", name=self.run_name, config=config or {})
        if config:
            with open(os.path.join(self.run_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

    def log_train(self, step: int, train_loss: float, lr: float, tokens_per_sec: float):
        wall = time.time() - self.run_start_time
        row = {
            "step": step,
            "wall_time_sec": wall,
            "train_loss": train_loss,
            "lr": lr,
            "tokens_per_sec": tokens_per_sec,
            "val_loss": "",
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        if self.use_wandb and self._wandb:
            self._wandb.log({"train/loss": train_loss, "lr": lr, "tokens_per_sec": tokens_per_sec, "step": step, "wall_time_sec": wall})

    def log_eval(self, step: int, val_loss: float):
        wall = time.time() - self.run_start_time
        row = {
            "step": step,
            "wall_time_sec": wall,
            "train_loss": "",
            "lr": "",
            "tokens_per_sec": "",
            "val_loss": val_loss,
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        if self.use_wandb and self._wandb:
            self._wandb.log({"valid/loss": val_loss, "step": step, "wall_time_sec": wall})

    def close(self):
        try:
            self._csv_file.close()
        except Exception:
            pass
        if self.use_wandb and self._wandb:
            try:
                self._wandb.finish()
            except Exception:
                pass
