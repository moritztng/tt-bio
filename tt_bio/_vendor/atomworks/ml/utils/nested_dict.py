"""Tools to work with nested dictionaries."""

from typing import Any


def _assert_dict_like(d: Any) -> None:
    """Assert that a value is a dictionary-like object."""
    if not hasattr(d, "items"):
        raise TypeError(f"Expected a dictionary-like object, got {type(d)}")


def flatten(d: dict[str, Any], *, fuse_keys: str | None = None) -> dict[tuple[str, ...], Any] | dict[str, Any]:
    """Flatten a nested dictionary into a single level dictionary with tuple keys, preserving non-dict values.

    Args:
        - d (dict): A nested dictionary to flatten.
        - fuse_keys (str | None): If provided, joins the key tuple elements with this string to create string keys.
            If None, returns tuple keys.

    Returns:
        - dict: A flattened dictionary where nested dict keys become either tuple keys or fused string keys,
            but other values remain intact.

    Example:
        >>> d = {"a": {"b": [1, 2]}, "c": {"d": {"e": 3}}, "f": [4, 5]}
        >>> flatten(d)
        {('a', 'b'): [1, 2], ('c', 'd', 'e'): 3, ('f',): [4, 5]}
        >>> flatten(d, fuse_keys=".")
        {'a.b': [1, 2], 'c.d.e': 3, 'f': [4, 5]}
    """
    _assert_dict_like(d)

    def _flatten(d: dict[str, Any], parent_key: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
        items: list[tuple[tuple[str, ...], Any]] = []
        for k, v in d.items():
            new_key = (*parent_key, str(k))
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flattened = _flatten(d)

    if fuse_keys is not None:
        return {fuse_keys.join(k): v for k, v in flattened.items()}
    return flattened


def unflatten(d: dict[tuple[str, ...] | str, Any], *, split_keys: str | None = None) -> dict[str, Any]:
    """Unflatten a flattened dictionary into a nested dictionary.

    Args:
        - d (dict): A flattened dictionary with either tuple keys or string keys.
        - split_keys (str | None): If provided, splits string keys with this string to create tuple keys.
            If None, expects tuple keys.

    Returns:
        - dict: A nested dictionary reconstructed from the flattened keys.

    Example:
        >>> d = {("a", "b"): [1, 2], ("c", "d", "e"): 3, ("f",): [4, 5]}
        >>> unflatten(d)
        {'a': {'b': [1, 2]}, 'c': {'d': {'e': 3}}, 'f': [4, 5]}
        >>> d = {"a.b": [1, 2], "c.d.e": 3, "f": [4, 5]}
        >>> unflatten(d, split_keys=".")
        {'a': {'b': [1, 2]}, 'c': {'d': {'e': 3}}, 'f': [4, 5]}
    """
    result: dict[str, Any] = {}

    for k, v in d.items():
        # Convert string keys to tuples if split_keys is provided
        keys = k.split(split_keys) if isinstance(k, str) and split_keys else k

        # Handle both string and tuple keys
        if not isinstance(keys, tuple | list):
            keys = (keys,)

        current = result
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = v

    return result


def get(d: dict[tuple[str, ...], Any], key: tuple[str, ...], default: Any = None) -> Any:
    """Get a value from a nested dictionary using a tuple key.

    Equivalent behavior to .get for nested dictionaries.

    Args:
        - d (dict): A nested dictionary.
        - key (tuple): A tuple of keys to navigate through the dictionary.
        - default (Any): The value to return if the key is not found.

    Returns:
        - Any: The value at the specified key. If the key is not found, the default value is returned.
    """
    _assert_dict_like(d)
    key = (key,) if isinstance(key, str) else key
    if len(key) == 0:
        raise KeyError("Empty key")

    for k in key:
        if k not in d:
            return default
        d = d[k]
    return d


def set(d: dict[tuple[str, ...], Any], key: tuple[str, ...], value: Any) -> None:
    """Set a value in a nested dictionary using a tuple key.

    Equivalent behavior to __setitem__ for nested dictionaries.
    Creates intermediate dictionaries if they don't exist yet.

    Args:
        - d (dict): A nested dictionary.
        - key (tuple): A tuple of keys to navigate through the dictionary.
        - value (Any): The value to set at the specified key.
    """
    _assert_dict_like(d)
    key = (key,) if isinstance(key, str) else key
    if len(key) == 0:
        raise KeyError("Empty key")

    current = d
    for k in key[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[key[-1]] = value


def getitem(d: dict[tuple[str, ...], Any], key: tuple[str, ...]) -> Any:
    """Get a value from a nested dictionary using a tuple key.

    Equivalent behavior to __getitem__ for nested dictionaries.

    Args:
        - d (dict): A nested dictionary.
        - key (tuple): A tuple of keys to navigate through the dictionary.

    Returns:
        - Any: The value at the specified key.
    """
    _assert_dict_like(d)
    key = (key,) if isinstance(key, str) else key
    if len(key) == 0:
        raise KeyError("Empty key")

    for k in key:
        if k not in d:
            raise KeyError(f"Level {k=} of nested {key=} not found in dictionary")
        d = d[k]
    return d
