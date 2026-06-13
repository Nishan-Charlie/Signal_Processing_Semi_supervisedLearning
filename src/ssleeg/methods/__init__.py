"""Semi-supervised learning methods.

Every method subclasses :class:`SSLMethod` and registers itself in ``METHODS``.
The trainer is method-agnostic: it calls ``compute_loss`` each step and
``on_step_end`` after the optimizer step, and asks the method which module to use
at evaluation time. This is the single extension point for adding *your own
proposed method* -- copy ``your_method.py`` and implement ``compute_loss``.

Categories implemented here:
* Consistency regularization : pi_model, mean_teacher, vat, ict
* Pseudo-labeling            : pseudo_label
* Modern SSL (matching)      : mixmatch, fixmatch, flexmatch
* Contrastive pretraining    : simclr
* Supervised baseline        : supervised
* Template                   : your_method
"""

from ssleeg.methods.base import SSLMethod, build_method, sigmoid_rampup, consistency_weight
from ssleeg.methods import (  # noqa: F401  (registers methods)
    supervised,
    pi_model,
    mean_teacher,
    vat,
    ict,
    pseudo_label,
    mixmatch,
    fixmatch,
    flexmatch,
    simclr,
    your_method,
)

__all__ = ["SSLMethod", "build_method", "sigmoid_rampup", "consistency_weight"]
