from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import logging
import tomllib

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "runtime.toml"


@dataclass(frozen=True)
class IngestConfig:
    anchor_date: Optional[str]
    window_months: int


@dataclass(frozen=True)
class FloraConfig:
    original_lookup_url: str
    cache_ttl_hours: int


@dataclass(frozen=True)
class RuntimeConfig:
    ingest: IngestConfig
    flora: FloraConfig


def _default_config() -> RuntimeConfig:
    return RuntimeConfig(
        ingest=IngestConfig(anchor_date=None, window_months=6),
        flora=FloraConfig(
            original_lookup_url="https://rep-api.forrt.org/v1/original-lookup",
            cache_ttl_hours=48,
        ),
    )


def _safe_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else fallback
    except Exception:
        return fallback


def load_runtime_config(config_path: Optional[Path] = None) -> RuntimeConfig:
    cfg = _default_config()
    path = config_path or _DEFAULT_CONFIG_PATH
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError:
        logger.warning("Runtime config file not found; using defaults", extra={"path": str(path)})
        return cfg
    except tomllib.TOMLDecodeError:
        logger.exception("Runtime config parse failed; using defaults", extra={"path": str(path)})
        return cfg
    except Exception:
        logger.exception("Runtime config load failed; using defaults", extra={"path": str(path)})
        return cfg

    ingest_raw = raw.get("ingest") if isinstance(raw, dict) else {}
    flora_raw = raw.get("flora") if isinstance(raw, dict) else {}
    if not isinstance(ingest_raw, dict):
        ingest_raw = {}
    if not isinstance(flora_raw, dict):
        flora_raw = {}

    anchor_date = ingest_raw.get("anchor_date", cfg.ingest.anchor_date)
    if isinstance(anchor_date, str):
        anchor_date = anchor_date.strip() or None
    elif anchor_date is None:
        anchor_date = None
    else:
        anchor_date = str(anchor_date).strip() or None

    window_months = _safe_int(ingest_raw.get("window_months", cfg.ingest.window_months), cfg.ingest.window_months)

    original_lookup_url = flora_raw.get("original_lookup_url", cfg.flora.original_lookup_url)
    if not isinstance(original_lookup_url, str) or not original_lookup_url.strip():
        original_lookup_url = cfg.flora.original_lookup_url
    else:
        original_lookup_url = original_lookup_url.strip()

    cache_ttl_hours = _safe_int(flora_raw.get("cache_ttl_hours", cfg.flora.cache_ttl_hours), cfg.flora.cache_ttl_hours)

    return RuntimeConfig(
        ingest=IngestConfig(anchor_date=anchor_date, window_months=window_months),
        flora=FloraConfig(
            original_lookup_url=original_lookup_url,
            cache_ttl_hours=cache_ttl_hours,
        ),
    )


RUNTIME_CONFIG = load_runtime_config()
