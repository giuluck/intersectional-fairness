from abc import abstractmethod
from typing import Dict, Any


class Item:
    """Interface for an item which can be part of the specification of an experiment."""

    @classmethod
    @abstractmethod
    def last_edit(cls) -> str:
        """A datetime string having format YYYY-MM-DD HH:MM:SS which represents the last time the code was edit."""
        pass

    @property
    @abstractmethod
    def configuration(self) -> Dict[str, Any]:
        """A dictionary of parameters which uniquely identifies the item."""
        pass

    def __eq__(self, other: Any) -> bool:
        if other is self:
            return True
        return isinstance(other, self.__class__) and self.configuration == other.configuration

    def __repr__(self):
        configuration = ', '.join([f'{key}={value}' for key, value in self.configuration.items()])
        return f"{self.__class__.__name__}({configuration})"
