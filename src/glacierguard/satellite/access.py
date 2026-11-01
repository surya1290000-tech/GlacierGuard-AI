"""Authenticated, cache-aware access to STAC raster assets.

Catalogue discovery deliberately does not imply download success.  This module
is the single boundary that obtains a bearer token and reads an asset.  It
never logs credentials, tokens, or complete signed URLs.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
from typing import Any, Dict, Optional
import urllib.error
import urllib.parse
import urllib.request

from glacierguard.config import Settings


class AssetAccessError(RuntimeError):
    """Raised when an imagery asset cannot be safely obtained."""


@dataclass
class AccessedAsset:
    key: str
    path: Path
    href_host: str
    byte_count: int
    sha256: str


class CDSEAssetClient:
    """Small OAuth client with token reuse, retries, and local cache reuse."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._token: Optional[str] = None
        self._expires_at = 0.0

    def _token_value(self) -> str:
        if self._token and time.time() < self._expires_at - 30:
            return self._token
        if not self.settings.client_id or not self.settings.client_secret:
            raise AssetAccessError("CDSE credentials are not configured for raw imagery access")
        body = urllib.parse.urlencode({
            "grant_type": "client_credentials", "client_id": self.settings.client_id,
            "client_secret": self.settings.client_secret,
        }).encode()
        request = urllib.request.Request(self.settings.token_url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(request, timeout=self.settings.asset_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise AssetAccessError(f"CDSE authentication unavailable: {type(exc).__name__}") from exc
        token = payload.get("access_token")
        if not token:
            raise AssetAccessError("CDSE authentication response did not contain an access token")
        self._token = str(token)
        self._expires_at = time.time() + int(payload.get("expires_in", 300))
        return self._token

    def fetch(self, product_id: str, key: str, asset: Dict[str, Any]) -> AccessedAsset:
        """Download one declared STAC asset once and verify it is non-empty."""
        href = asset.get("href") if isinstance(asset, dict) else None
        if not href or not isinstance(href, str) or not href.startswith(("https://", "http://", "file://")):
            raise AssetAccessError(f"Asset {key!r} has no usable href")
        suffix = Path(urllib.parse.urlparse(href).path).suffix or ".bin"
        safe_product = hashlib.sha256(product_id.encode()).hexdigest()[:16]
        destination = self.settings.processing_cache_dir / safe_product / f"{key}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size > 0:
            content = destination.read_bytes()
            return AccessedAsset(key, destination, urllib.parse.urlparse(href).netloc, len(content), hashlib.sha256(content).hexdigest())

        last_error: Optional[Exception] = None
        for attempt in range(self.settings.stac_max_retries):
            try:
                headers = {"User-Agent": "GlacierGuard-AI/phase8"}
                if not href.startswith("file://"):
                    headers["Authorization"] = f"Bearer {self._token_value()}"
                request = urllib.request.Request(href, headers=headers)
                with urllib.request.urlopen(request, timeout=self.settings.asset_timeout_seconds) as response:
                    content = response.read()
                if not content:
                    raise AssetAccessError(f"Asset {key!r} download was empty")
                destination.write_bytes(content)
                return AccessedAsset(key, destination, urllib.parse.urlparse(href).netloc, len(content), hashlib.sha256(content).hexdigest())
            except (urllib.error.URLError, OSError, AssetAccessError) as exc:
                last_error = exc
                if attempt + 1 < self.settings.stac_max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    self._token = None  # retry may require a refreshed token
        # The underlying message is intentionally categorical and contains no
        # token, header, or signed URL; retaining it makes an unavailable
        # credential distinguishable from a transient download failure.
        raise AssetAccessError(f"Unable to access asset {key!r}: {last_error}") from last_error
