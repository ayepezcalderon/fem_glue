import json
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

_CONFIG_FILE_NAME = "femglue.json"


class _Configuration(BaseModel):
    """Singleton containing default configuration parameters.

    Example parameter is the floating point precision of mathematical operations.

    If femglue.json is defined in the current working directory,
    this configuration defines the values of this configuration singleton.
    Otherwise, the default values of the singleton are used.

    Attributes
    ----------
    precision : int
        The number of decimal places for floats in the package.
        By default 6.
    tol : float
        The inverse of precision.
        Quantities smaller than this one are considered to be approximately zero.
        By default 1e-6

    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    precision: int = Field(default=6)

    def __new__(cls) -> Self:
        """Make the class a singleton.

        Use the static configuration file to create the singleton.

        Returns
        -------
        Self
            Singleton configuration.

        """
        if not hasattr(cls, "_instance"):
            cls._instance = BaseModel.__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs) -> None:
        """Make sure that the singleton is only initialized once."""
        if not hasattr(self.__class__, "_initialized"):
            # Merge static config if it exists
            try:
                with open(_CONFIG_FILE_NAME) as f:
                    static_config = json.load(f)

                if not isinstance(static_config, dict):
                    raise TypeError(
                        f"'{_CONFIG_FILE_NAME}' must be defined as a dictionary."
                    )

                kwargs |= static_config
            except FileNotFoundError:
                ...

            # Initialize base model
            super().__init__(*args, **kwargs)

            # Mark as initialized
            _Configuration._initialized = True

    @property
    def tol(self):
        return 10**-self.precision


CONFIG = _Configuration()
