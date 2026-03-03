from __future__ import annotations


def invalidate_canonical_mappings_cache(
    increment_db_version: bool = True,
) -> None:
    try:
        from server import _invalidate_canonical_mappings_cache

        _invalidate_canonical_mappings_cache(increment_db_version=increment_db_version)
    except Exception:
        try:
            from database import (
                get_canonical_mappings_cache,
                increment_canonical_mappings_cache_version,
            )

            if increment_db_version:
                increment_canonical_mappings_cache_version()
            get_canonical_mappings_cache().clear()
        except Exception:
            return
