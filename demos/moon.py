#!/usr/bin/env python3

import argparse
import logging
import math
import os
import random
import sys
from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from micrograd.engine import Value
from micrograd.nn import MLP


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("trainer")
    logger.propagate = False  # avoid duplicate handlers
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    lvl = getattr(logging, level.upper(), logging.INFO)

    try:
        # Prefer Rich if present (nicer tracebacks + levels)
        from rich.logging import RichHandler  # type: ignore

        handler = RichHandler(
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        fmt = "%(message)s"
        datefmt = "[%X]"
        logger.addHandler(handler)
        logger.setLevel(lvl)
        logging.basicConfig(level=lvl, format=fmt, datefmt=datefmt, handlers=[handler])
        logger.debug("Using RichHandler for logging.")
    except Exception:
        # Fallback to standard logging with concise format
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(lvl)
        logger.debug("Using standard logging handler.")

    return logger


def parse_hidden_sizes(s: str) -> List[int]:
    """
    Parse a comma/space separated list of integers, e.g. "16,16,1" or "16 16 1".
    """
    parts = [p for chunk in s.replace(",", " ").split() for p in [chunk.strip()] if p]
    if not parts:
        raise argparse.ArgumentTypeError("hidden sizes cannot be empty")
    try:
        hs = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"could not parse hidden sizes from {s!r}")
    if any(h <= 0 for h in hs):
        raise argparse.ArgumentTypeError("all hidden sizes must be positive")
    return hs


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def build_dataset(
    n_samples: int, noise: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng_state = np.random.get_state()
    set_seeds(seed)
    X, y = make_moons(n_samples=n_samples, noise=noise)
    np.random.set_state(
        rng_state
    )  # keep global RNG sequence consistent after dataset gen
    # map labels from {0,1} -> {-1, +1}
    y = y * 2 - 1
    return X, y


def accuracy_from_scores(yb: Iterable[int], scores: Iterable[Value]) -> float:
    # sign agreement in {−1,+1}
    acc = [(yi > 0) == (si.data > 0) for yi, si in zip(yb, scores)]
    return float(sum(acc)) / float(len(acc))


def train(
    X: np.ndarray,
    y: np.ndarray,
    hidden_sizes: List[int],
    epochs: int,
    batch_size: Optional[int],
    alpha: float,
    log_every: int,
    logger: logging.Logger,
) -> Tuple[MLP, float, float]:
    """
    Train MLP with hinge loss + L2 reg using SGD and a linear LR schedule.
    Returns (model, final_loss, final_acc).
    """
    input_dim = X.shape[1]
    model = MLP(input_dim, hidden_sizes)

    logger.info(str(model))
    logger.info("number of parameters: %d", len(model.parameters()))

    # closure uses X, y, model by reference
    def loss_fn(batch_size: Optional[int] = None) -> Tuple[Value, float]:
        if batch_size is None:
            Xb, yb = X, y
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = X[ri], y[ri]

        inputs = [list(map(Value, xrow)) for xrow in Xb]
        scores = list(map(model, inputs))

        # SVM hinge loss: mean(max(0, 1 - y*f(x)))
        losses = [(1 + -yi * si).relu() for yi, si in zip(yb, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))

        # L2 regularization on parameters
        reg_loss = alpha * sum((p * p for p in model.parameters()))

        total = data_loss + reg_loss
        acc = accuracy_from_scores(yb, scores)
        return total, acc

    # initial evaluation
    total_loss, acc = loss_fn(batch_size=None)
    logger.info("init: loss=%.6f, acc=%.2f%%", total_loss.data, acc * 100.0)

    # optimization
    for k in range(epochs):
        total_loss, acc = loss_fn(batch_size=batch_size)

        model.zero_grad()
        total_loss.backward()

        # linear anneal from 1.0 -> 0.1 over epochs (as original 1.0 - 0.9*k/100 with epochs=100)
        lr = 1.0 - 0.9 * (k / max(1, epochs))
        for p in model.parameters():
            p.data -= lr * p.grad

        if (k % max(1, log_every)) == 0:
            logger.info(
                "step %d | lr=%.4f | loss=%.6f | acc=%.2f%%",
                k,
                lr,
                total_loss.data,
                acc * 100.0,
            )

    return model, float(total_loss.data), float(acc)


def plot_results(model: MLP, X: np.ndarray, y: np.ndarray, grid_step: float) -> None:
    # scatter of data
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap="jet")
    plt.title("Data (make_moons)")
    plt.show()

    # decision boundary
    h = grid_step
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(Value, xrow)) for xrow in Xmesh]
    scores = list(map(model, inputs))
    Z = np.array([s.data > 0 for s in scores])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary")
    plt.show()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a micrograd MLP with hinge loss on make_moons (argparse + pretty logging).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    p.add_argument(
        "--n-samples", type=int, default=100, help="Number of samples for make_moons"
    )
    p.add_argument(
        "--noise", type=float, default=0.1, help="Noise level for make_moons"
    )
    p.add_argument("--seed", type=int, default=1337, help="PRNG seed (numpy & random)")

    # model
    p.add_argument(
        "--hidden-sizes",
        type=parse_hidden_sizes,
        default=parse_hidden_sizes("16,16,1"),
        help='Layer sizes for MLP after the input dim, e.g. "16,16,1"',
    )

    # optimization
    p.add_argument(
        "--epochs", type=int, default=100, help="Number of SGD steps (was 100)"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size. Omit for full batch (original behavior).",
    )
    p.add_argument(
        "--alpha", type=float, default=1e-4, help="L2 regularization coefficient"
    )
    p.add_argument(
        "--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, ...)"
    )
    p.add_argument("--log-every", type=int, default=1, help="Log every k steps")

    # plotting
    p.add_argument(
        "--grid-step", type=float, default=0.25, help="Grid step for decision boundary"
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plotting (useful for headless or benchmarking runs)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    logger = setup_logger(args.log_level)

    set_seeds(args.seed)
    X, y = build_dataset(n_samples=args.n_samples, noise=args.noise, seed=args.seed)

    model, final_loss, final_acc = train(
        X=X,
        y=y,
        hidden_sizes=args.hidden_sizes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        log_every=args.log_every,
        logger=logger,
    )
    logger.info("final: loss=%.6f, acc=%.2f%%", final_loss, final_acc * 100.0)

    if not args.no_plots:
        plot_results(model, X, y, grid_step=args.grid_step)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
