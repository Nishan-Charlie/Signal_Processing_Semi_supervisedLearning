"""A minimal, type-safe registry for plug-and-play components.

The registry pattern is what makes the framework *extensible*: datasets, model
backbones, and SSL methods all register themselves under a string name and can
then be instantiated purely from a config file. Adding a new component (for
example, *your own proposed SSL method*) is as simple as decorating a class with
``@METHODS.register("my_method")`` -- no other file needs to change.
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterable, List, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> object mapping with a decorator-based registration API."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, T] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, key: str | None = None) -> Callable[[T], T]:
        """Decorator that registers a class/function under ``key``.

        If ``key`` is omitted, the object's ``__name__`` (lower-cased) is used.
        """

        def decorator(obj: T) -> T:
            name = key if key is not None else getattr(obj, "__name__", None)
            if name is None:
                raise ValueError("Could not infer a registry key; pass one explicitly.")
            name = name.lower()
            if name in self._registry:
                raise KeyError(
                    f"'{name}' is already registered in registry '{self._name}'. "
                    f"Existing: {self._registry[name]!r}"
                )
            self._registry[name] = obj
            return obj

        return decorator

    def get(self, key: str) -> T:
        key = key.lower()
        if key not in self._registry:
            raise KeyError(
                f"'{key}' not found in registry '{self._name}'. "
                f"Available: {sorted(self._registry)}"
            )
        return self._registry[key]

    def build(self, key: str, *args, **kwargs):
        """Instantiate the registered object with the given arguments."""
        return self.get(key)(*args, **kwargs)

    def keys(self) -> List[str]:
        return sorted(self._registry)

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._registry

    def __iter__(self) -> Iterable[str]:
        return iter(self.keys())

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, entries={self.keys()})"


# Global registries used throughout the framework.
DATASETS: "Registry" = Registry("datasets")
MODELS: "Registry" = Registry("models")
METHODS: "Registry" = Registry("methods")
