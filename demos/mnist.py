#!/usr/bin/env python3
"""
Train a micrograd MLP on MNIST with multi-class hinge loss (+ L2).
Argparse + pretty logging. Defaults sized for demo speed rather than SOTA accuracy.

Math:
  For sample i with label y_i and scores s \in R^{10}:
    L_i = sum_{j!=y_i} max(0, 1 + s_j - s_{y_i})
  Total: (1/|B|) * sum_i L_i + alpha * ||theta||_2^2
"""

import argparse
import logging
import random
import sys
from typing import Iterable, List, Optional, Tuple

import numpy as np

# plotting only for sanity-check visualization of a few digits (no decision boundary here)
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml, load_digits

from micrograd.engine import Value
from micrograd.nn import MLP


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("mnist_micrograd")
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    lvl = getattr(logging, level.upper(), logging.INFO)
    try:
        from rich.logging import RichHandler  # type: ignore

        handler = RichHandler(
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        fmt, datefmt = "%(message)s", "[%X]"
        logger.addHandler(handler)
        logger.setLevel(lvl)
        logging.basicConfig(level=lvl, format=fmt, datefmt=datefmt, handlers=[handler])
    except Exception:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(lvl)
    return logger


def parse_hidden_sizes(s: str) -> List[int]:
    parts = [p for chunk in s.replace(",", " ").split() for p in [chunk.strip()] if p]
    hs = [int(p) for p in parts]
    if hs[-1] != 10:
        raise argparse.ArgumentTypeError(
            "last hidden size must be 10 (number of classes)"
        )
    if any(h <= 0 for h in hs):
        raise argparse.ArgumentTypeError("all layer sizes must be positive")
    return hs


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def load_dataset(
    which: str,
    limit_train: Optional[int],
    limit_test: Optional[int],
    seed: int,
    logger: logging.Logger,
):
    if which == "mnist":
        try:
            logger.info("Loading MNIST (70k, 28x28) from OpenML...")
            X, y = fetch_openml(
                "mnist_784",
                version=1,
                as_frame=False,
                parser="auto",
                cache=True,
                data_home=None,
                return_X_y=True,
            )
            X = X.astype(np.float32) / 255.0
            y = y.astype(np.int64)
            n = X.shape[0]
            # stratified split by simple per-class slicing after a shuffle
            idx = np.arange(n)
            rng = np.random.RandomState(seed)
            rng.shuffle(idx)
            X, y = X[idx], y[idx]
            # standard split: 60k train, 10k test (post-shuffle)
            Xtr, Ytr = X[:60000], y[:60000]
            Xte, Yte = X[60000:], y[60000:]
        except Exception as e:
            logger.warning(
                "Falling back to sklearn.load_digits (8x8) because MNIST load failed: %s",
                e,
            )
            digits = load_digits()
            X = digits.data.astype(np.float32) / 16.0  # already 0..16
            y = digits.target.astype(np.int64)
            n = X.shape[0]
            idx = np.arange(n)
            rng = np.random.RandomState(seed)
            rng.shuffle(idx)
            X, y = X[idx], y[idx]
            # 80/20 split
            cut = int(0.8 * n)
            Xtr, Ytr = X[:cut], y[:cut]
            Xte, Yte = X[cut:], y[cut:]
    else:
        raise ValueError("unknown dataset")

    # optional subsetting for speed
    if limit_train is not None:
        Xtr, Ytr = Xtr[:limit_train], Ytr[:limit_train]
    if limit_test is not None:
        Xte, Yte = Xte[:limit_test], Yte[:limit_test]

    logger.info(
        "Train set: %s, Test set: %s, Dim: %d", Xtr.shape, Xte.shape, Xtr.shape[1]
    )
    return Xtr, Ytr, Xte, Yte


def multiclass_hinge_loss(scores: List[Value], yi: int) -> Value:
    s_y = scores[yi]
    margins = [(Value(1.0) + s - s_y).relu() for j, s in enumerate(scores) if j != yi]
    return sum(margins)


def accuracy_from_scores_batch(
    yb: Iterable[int], score_lists: Iterable[List[Value]]
) -> float:
    preds = [int(np.argmax([s.data for s in scores])) for scores in score_lists]
    yb = list(yb)
    return float(np.mean(np.array(preds) == np.array(yb)))


def train_epoch(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    alpha: float,
    lr: float,
) -> Tuple[float, float]:
    n = X.shape[0]
    perm = np.random.permutation(n)
    # accumulate running loss/acc for logging
    total_loss_value = 0.0
    total_correct = 0
    seen = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = perm[start:end]
        Xb, yb = X[idx], y[idx]

        # forward
        inputs = [list(map(Value, xrow)) for xrow in Xb]  # each xrow -> list[Value]
        score_lists = [model(xv) for xv in inputs]  # each -> list[Value] of length 10

        # data loss (mean over batch)
        losses = [
            multiclass_hinge_loss(scores, int(yi))
            for scores, yi in zip(score_lists, yb)
        ]
        data_loss = sum(losses) * (1.0 / len(losses))

        # L2 reg
        reg_loss = alpha * sum((p * p for p in model.parameters()))

        total = data_loss + reg_loss

        # backward/update
        model.zero_grad()
        total.backward()
        for p in model.parameters():
            p.data -= lr * p.grad

        # metrics
        acc = accuracy_from_scores_batch(yb, score_lists)
        total_loss_value += float(total.data) * len(yb)
        total_correct += int(acc * len(yb))
        seen += len(yb)

    return total_loss_value / seen, total_correct / seen


def evaluate(
    model: MLP, X: np.ndarray, y: np.ndarray, batch_size: int = 512
) -> Tuple[float, float]:
    n = X.shape[0]
    total_loss_value = 0.0
    total_correct = 0
    seen = 0
    alpha = 0.0  # no reg at eval

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb, yb = X[start:end], y[start:end]
        inputs = [list(map(Value, xrow)) for xrow in Xb]
        score_lists = [model(xv) for xv in inputs]
        losses = [
            multiclass_hinge_loss(scores, int(yi))
            for scores, yi in zip(score_lists, yb)
        ]
        data_loss = sum(losses) * (1.0 / len(losses))
        reg_loss = alpha * sum((p * p for p in model.parameters()))
        total = data_loss + reg_loss

        acc = accuracy_from_scores_batch(yb, score_lists)
        total_loss_value += float(total.data) * len(yb)
        total_correct += int(acc * len(yb))
        seen += len(yb)

    return total_loss_value / seen, total_correct / seen


def build_argparser():
    p = argparse.ArgumentParser(
        description="micrograd MLP on MNIST with multi-class hinge loss",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    p.add_argument(
        "--dataset", choices=["mnist"], default="mnist", help="Dataset to use"
    )
    p.add_argument(
        "--limit-train",
        type=int,
        default=20000,
        help="Limit training examples for speed (None for all)",
    )
    p.add_argument(
        "--limit-test",
        type=int,
        default=5000,
        help="Limit test examples for speed (None for all)",
    )
    p.add_argument("--seed", type=int, default=1337, help="Random seed")
    # model
    p.add_argument(
        "--hidden-sizes",
        type=parse_hidden_sizes,
        default=[128, 64, 10],
        help='Layer sizes after input dim; last must be 10, e.g. "128,64,10"',
    )
    # optimization
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs (each sees all selected train data once)",
    )
    p.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    p.add_argument(
        "--alpha", type=float, default=5e-5, help="L2 regularization coefficient"
    )
    p.add_argument("--lr-start", type=float, default=1.0, help="Starting learning rate")
    p.add_argument(
        "--lr-end",
        type=float,
        default=0.1,
        help="Ending learning rate (linear schedule)",
    )
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    p.add_argument("--log-every", type=int, default=1, help="Log every k epochs")
    # viz
    p.add_argument(
        "--show-samples", action="store_true", help="Show a small grid of sample digits"
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    logger = setup_logger(args.log_level)
    set_seeds(args.seed)

    Xtr, Ytr, Xte, Yte = load_dataset(
        which=args.dataset,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        seed=args.seed,
        logger=logger,
    )
    nin = Xtr.shape[1]
    if args.hidden_sizes[0] <= 0 or args.hidden_sizes[-1] != 10:
        raise SystemExit("hidden sizes must end with 10")

    # model
    model = MLP(nin, args.hidden_sizes)
    logger.info("Model: %s", model)
    logger.info("Params: %d", len(model.parameters()))

    # optional sanity-check viz
    side = int(np.sqrt(nin))
    fig, axes = plt.subplots(2, 5, figsize=(8, 3))
    for ax, i in zip(axes.flat, range(10)):
        ax.imshow(Xtr[i].reshape(side, side), cmap="gray")
        ax.set_title(f"y={int(Ytr[i])}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # training loop
    for epoch in range(args.epochs):
        # linear LR schedule from lr_start -> lr_end
        t = epoch / max(1, args.epochs - 1)
        lr = (1 - t) * args.lr_start + t * args.lr_end

        tr_loss, tr_acc = train_epoch(
            model=model,
            X=Xtr,
            y=Ytr,
            batch_size=args.batch_size,
            alpha=args.alpha,
            lr=lr,
        )
        te_loss, te_acc = evaluate(model, Xte, Yte)

        if epoch % max(1, args.log_every) == 0:
            logging.info(
                "epoch %d | lr=%.4f | train loss=%.6f acc=%.2f%% | test loss=%.6f acc=%.2f%%",
                epoch,
                lr,
                tr_loss,
                tr_acc * 100.0,
                te_loss,
                te_acc * 100.0,
            )

    te_loss, te_acc = evaluate(model, Xte, Yte)
    logger.info("FINAL | test loss=%.6f | test acc=%.2f%%", te_loss, te_acc * 100.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
