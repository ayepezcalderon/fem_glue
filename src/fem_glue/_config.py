import json
from typing import Self
from pydantic import BaseModel, Field

_CONFIG_FILE_NAME = "femglue.json"


class _Configuration(BaseModel):
    precision: int = Field(default=6)

    def __new__(cls, *args, **kwargs) -> Self:
        """
        Use the static configuration file to create the singleton.
        Make the class a singleton.

        Returns
        -------
        Self
            Singleton configuration.
        """
        if not hasattr(cls, "_instance"):
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

            cls._instance = BaseModel.__new__(cls, *args, **kwargs)

        return cls._instance

    def __init__(self, *args, **kwargs) -> None:
        """
        Make sure that the singleton is only initialized once.
        """
        if not hasattr(self.__class__, "_initialized"):
            super().__init__(*args, **kwargs)
            _Configuration._initialized = True

    @property
    def tol(self):
        return 10**-self.precision


CONFIG = _Configuration()
