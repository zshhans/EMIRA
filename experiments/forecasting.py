from dataclasses import asdict
import json
from pathlib import Path
import statistics

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from config.emira_config import DATASETS, EMIRAConfig, dataset_config
from data_modules.utils import get_dataloaders
from models.determinism import configure_determinism
from tasks.fcst_module import ForecastingModule


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def execution(args, enabled):
    configure_determinism(enabled)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Use --device cpu for the reference implementation."
            )
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        except ImportError as exc:
            raise RuntimeError(
                "Install mamba_ssm==2.2.5 for CUDA execution; see README.md."
            ) from exc
        return dict(accelerator="gpu", devices=[device.index or 0])
    if device.type != "cpu":
        raise ValueError("--device must be cpu or cuda:<index>")
    return dict(accelerator="cpu", devices=1)


def fit(config, args, trainer_device):
    logger = CSVLogger(args.output_dir, name=f"{config.dataset}/seed{config.seed}")
    run_dir = Path(logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = get_dataloaders(
        config, args.num_workers, run_dir / "sampler_orders.jsonl"
    )
    write_json(run_dir / "config.json", asdict(config))
    pl.seed_everything(config.seed, workers=True)
    module = ForecastingModule(config)
    checkpoint = ModelCheckpoint(
        monitor="val_mse",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_mse:.6f}",
    )
    trainer = pl.Trainer(
        **trainer_device,
        logger=logger,
        callbacks=[
            checkpoint,
            EarlyStopping(monitor="val_mse", mode="min", patience=6),
        ],
        max_epochs=config.max_epochs,
        precision="32-true",
        deterministic=config.deterministic,
        benchmark=not config.deterministic,
        gradient_clip_val=config.grad_clip_val,
    )
    trainer.fit(module, train, val)
    metrics = trainer.test(module, dataloaders=test, ckpt_path="best")[0]
    result = dict(
        dataset=config.dataset,
        seed=config.seed,
        best_val_mse=float(checkpoint.best_model_score),
        checkpoint=str(Path(checkpoint.best_model_path).resolve()),
        **metrics,
    )
    write_json(run_dir / "test.json", result)
    return result


def run_forecasting(args):
    trainer_device = execution(args, args.deterministic)
    datasets = DATASETS if "all" in args.dataset else tuple(dict.fromkeys(args.dataset))
    summary_logger = CSVLogger(args.output_dir, name="summary")
    summary_dir = Path(summary_logger.log_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset in datasets:
        for seed in dict.fromkeys(args.seeds):
            config = dataset_config(dataset, args.data_root, seed, args.deterministic)
            results.append(fit(config, args, trainer_device))
            write_json(summary_dir / "runs.json", results)
    summary = {}
    for dataset in datasets:
        rows = [row for row in results if row["dataset"] == dataset]
        summary[dataset] = dict(n=len(rows), seeds=[row["seed"] for row in rows])
        for metric in ("test_mse", "test_mae"):
            values = [row[metric] for row in rows]
            summary[dataset][metric] = dict(
                mean=statistics.mean(values),
                sample_std=statistics.stdev(values) if len(values) > 1 else None,
            )
    write_json(summary_dir / "aggregate.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Results: {summary_dir.resolve()}")


def run_evaluation(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = EMIRAConfig(**checkpoint["hyper_parameters"])
    config.datapath = dataset_config(config.dataset, args.data_root).datapath
    if args.deterministic is not None:
        config.deterministic = args.deterministic
    trainer_device = execution(args, config.deterministic)
    _, _, test = get_dataloaders(config, args.num_workers)
    pl.seed_everything(config.seed, workers=True)
    module = ForecastingModule(config)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    logger = CSVLogger(
        args.output_dir, name=f"evaluation/{config.dataset}/seed{config.seed}"
    )
    trainer = pl.Trainer(
        **trainer_device,
        logger=logger,
        precision="32-true",
        deterministic=config.deterministic,
        benchmark=not config.deterministic,
    )
    metrics = trainer.test(module, dataloaders=test)[0]
    write_json(Path(logger.log_dir) / "config.json", asdict(config))
    write_json(
        Path(logger.log_dir) / "test.json",
        dict(
            dataset=config.dataset,
            seed=config.seed,
            checkpoint=str(Path(args.checkpoint).resolve()),
            **metrics,
        ),
    )
