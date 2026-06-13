"""ssleeg: a reproducible benchmark framework for semi-supervised learning on EEG emotion recognition."""

__version__ = "0.1.0"

# Importing the package registers all components (datasets, models, methods) via decorators.
from ssleeg import data, models, methods  # noqa: F401,E402

__all__ = ["__version__", "data", "models", "methods"]
