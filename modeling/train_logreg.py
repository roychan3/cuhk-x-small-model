#!/usr/bin/env python3
"""Compatibility entry point for the logistic-regression algorithm."""

from modeling.train import main


if __name__ == "__main__":
    raise SystemExit(main(default_algorithm="logistic_regression"))
