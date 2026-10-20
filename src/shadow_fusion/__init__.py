"""Image compositing only: no removal model or language model is loaded."""
from .pipeline import Config, compose, make_config

__all__ = ["Config", "compose", "make_config"]
__version__ = "0.1.0"
