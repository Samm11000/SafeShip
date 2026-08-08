"""
Training datasets for the base model.

`apachejit` is real labelled commits and is what the base model should be
trained on. `ml/generate_synthetic.py` remains for the cold-start story, but
a model fitted to it has only learned the rule that produced it.
"""
from . import apachejit  # noqa: F401

__all__ = ["apachejit"]
