"""
.. include:: ../README.md
"""
from importlib.metadata import version
from typing import Optional, TypeVar

__version__ = version(__name__)

T = TypeVar("T")


def unwrap(value: Optional[T]) -> T:
    """Simple unwrap method to read datasets that are assumed to have data

    Example:
    ```python
    dataset.write(pl.DataFrame(...))
    unwrap(dataset.scan()).collect() # Should not raise
    ```

    Raises:
        ValueError: If value is None
    """
    if value is None:
        raise ValueError("Value is None")
    return value
