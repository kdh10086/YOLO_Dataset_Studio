
import os
from collections.abc import Mapping, Sequence
from typing import Dict, Iterable, List, Tuple


def get_label_path(image_path):
    """Finds the corresponding label file for a given image file in a robust way."""
    parts = image_path.split(os.sep)
    try:
        # Find the last occurrence of 'images' and replace it with 'labels'
        idx = len(parts) - 1 - parts[::-1].index('images')
        parts[idx] = 'labels'
        label_path_base, _ = os.path.splitext(os.sep.join(parts))
        return label_path_base + '.txt'
    except ValueError:
        # Fallback if 'images' is not in the path, though less common
        return os.path.splitext(image_path)[0] + '.txt'


def _next_available_id(used_ids: set, start_at: int = 0) -> int:
    """Helper that returns the next unused integer identifier."""
    candidate = start_at
    while candidate in used_ids:
        candidate += 1
    used_ids.add(candidate)
    return candidate


def normalize_class_map(raw_classes) -> Dict[int, str]:
    """Return a deterministic mapping of class IDs to names.

    Supports dictionaries with string or integer keys as well as iterable
    collections of class names. When the input does not specify explicit
    numeric identifiers, sequential IDs are assigned starting from zero.
    """
    if not raw_classes:
        return {}

    normalized: Dict[int, str] = {}
    used_ids: set = set()

    if isinstance(raw_classes, Mapping):
        auto_start = 0
        for key, name in raw_classes.items():
            idx = None
            if isinstance(key, int):
                idx = key
            else:
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    idx = None

            if idx is None or idx in used_ids:
                idx = _next_available_id(used_ids, auto_start)
                auto_start = idx + 1
            else:
                used_ids.add(idx)
                auto_start = max(auto_start, idx + 1)

            normalized[idx] = str(name)

        return normalized

    if isinstance(raw_classes, Sequence) and not isinstance(raw_classes, (str, bytes)):
        for name in raw_classes:
            idx = _next_available_id(used_ids)
            normalized[idx] = str(name)
        return normalized

    if isinstance(raw_classes, Iterable) and not isinstance(raw_classes, (str, bytes)):
        for name in raw_classes:
            idx = _next_available_id(used_ids)
            normalized[idx] = str(name)
        if normalized:
            return normalized

    # Unsupported type – return empty mapping to signal fallback behaviour
    return {}


def build_class_hotkeys(class_map: Dict[int, str], max_keys: int = 9) -> List[Tuple[int, int, str]]:
    """Create deterministic hotkey mappings for class selection.

    Returns a list of (hotkey_number, class_id, class_name) tuples ordered by
    the insertion order of ``class_map``. Only ``max_keys`` entries are
    produced so the first nine classes map to number keys 1-9.
    """
    normalized = normalize_class_map(class_map)
    if not normalized and class_map:
        if isinstance(class_map, Mapping):
            normalized = normalize_class_map(list(class_map.values()))
        else:
            normalized = normalize_class_map(list(class_map))
    hotkeys: List[Tuple[int, int, str]] = []

    for offset, (class_id, name) in enumerate(normalized.items(), start=1):
        if offset > max_keys:
            break
        hotkeys.append((offset, class_id, name))

    return hotkeys
