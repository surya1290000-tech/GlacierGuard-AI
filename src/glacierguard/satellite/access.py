"""
Authenticated, Cache-Aware and Protocol-Aware Access to Satellite Assets
========================================================================
Catalogue discovery deliberately does not imply download success. This module
is the single boundary that accesses raster assets.

Supported protocols:
- ``file://``: Direct local file access for fixtures, offline caches, and tests.
- ``https://`` / ``http://``: CDSE OAuth Bearer-authenticated HTTPS downloads.
- ``s3://``: Copernicus Data Space Ecosystem S3 object access (e.g., ``s3://eodata/...``).

Requirements:
- Credentials loaded strictly from environment variables (or settings).
- Safe token refresh upon expiration.
- Never logs credentials, authorization headers, access tokens, or signed URLs.
- Retries with exponential backoff and timeouts.
- Content verification and raster readability validation.
- Never silently rewrites unsupported URLs.
"""

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional
import urllib.error
import urllib.parse
import urllib.request

from glacierguard.config import Settings, default_settings


logger = logging.getLogger(__name__)


class AssetAccessError(RuntimeError):
    """Raised when an imagery asset cannot be safely accessed or verified."""


@dataclass
class AccessedAsset:
    key: str
    path: Path
    href_host: str
    byte_count: int
    sha256: str


class CDSEAssetClient:
    """Protocol-aware asset client with token reuse, retries, and local cache reuse."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or default_settings
        self._token: Optional[str] = None
        self._expires_at = 0.0

    def _token_value(self) -> str:
        """Obtain or refresh CDSE OAuth bearer token."""
        if self._token and time.time() < self._expires_at - 30:
            return self._token

        client_id = self.settings.client_id
        client_secret = self.settings.client_secret

        if not client_id or not client_secret:
            raise AssetAccessError(
                "CDSE credentials are not configured for raw imagery access "
                "(set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET)"
            )

        body = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }).encode("utf-8")

        request = urllib.request.Request(
            self.settings.token_url,
            data=body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "GlacierGuard-AI/phase8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.asset_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise AssetAccessError(
                f"CDSE authentication unavailable: {type(exc).__name__}"
            ) from exc

        token = payload.get("access_token")
        if not token:
            raise AssetAccessError(
                "CDSE authentication response did not contain an access token"
            )

        self._token = str(token)
        self._expires_at = time.time() + int(payload.get("expires_in", 300))
        return self._token

    def _validate_raster(self, path: Path, key: str) -> None:
        """Verify that a downloaded or cached file is a valid, readable raster."""
        try:
            import rasterio
            with rasterio.open(path) as src:
                if src.width <= 0 or src.height <= 0:
                    raise AssetAccessError(f"Raster {key!r} has invalid dimensions")
                # Ensure at least profile/metadata can be inspected
                _ = src.count
        except Exception as exc:
            # Clean up corrupted file so subsequent attempts don't hit bad cache
            path.unlink(missing_ok=True)
            raise AssetAccessError(
                f"Asset {key!r} is not a valid readable raster: {type(exc).__name__}"
            ) from exc

    def _fetch_file_url(self, href: str, destination: Path, key: str) -> AccessedAsset:
        """Access a local file:// URL."""
        parsed = urllib.parse.urlparse(href)
        # Handle file:///C:/path (Windows) or file:///path (Unix)
        local_path = urllib.request.url2pathname(parsed.path)
        src = Path(local_path)
        if not src.is_file() or src.stat().st_size == 0:
            raise AssetAccessError(f"Local asset file does not exist or is empty: {href}")

        self._validate_raster(src, key)
        content = src.read_bytes()
        return AccessedAsset(
            key=key,
            path=src,
            href_host="localhost",
            byte_count=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )

    def _fetch_s3_url(self, href: str, destination: Path, key: str) -> AccessedAsset:
        """Access a Copernicus Data Space Ecosystem s3:// URL."""
        s3_access_key = os.getenv("CDSE_S3_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID")
        s3_secret_key = os.getenv("CDSE_S3_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
        s3_endpoint = os.getenv(
            "CDSE_S3_ENDPOINT_URL", "https://eodata.dataspace.copernicus.eu"
        )

        if not s3_access_key or not s3_secret_key:
            raise AssetAccessError(
                f"CDSE S3 credentials are not configured for S3 imagery access "
                f"(set CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY to access {href})"
            )

        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise AssetAccessError(
                "boto3 is required to access CDSE s3:// asset references"
            ) from exc

        parsed = urllib.parse.urlparse(href)
        bucket = parsed.netloc
        s3_key = parsed.path.lstrip("/")

        session = boto3.session.Session()
        s3_client = session.client(
            service_name="s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            config=Config(s3={"addressing_style": "virtual"}),
        )

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(bucket, s3_key, str(destination))
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise AssetAccessError(
                f"Failed to download asset {key!r} from S3 bucket {bucket!r}: {type(exc).__name__}"
            ) from exc

        self._validate_raster(destination, key)
        content = destination.read_bytes()
        return AccessedAsset(
            key=key,
            path=destination,
            href_host=bucket,
            byte_count=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )

    def _fetch_http_url(self, href: str, destination: Path, key: str) -> AccessedAsset:
        """Download an HTTP/HTTPS asset with OAuth bearer token and retries."""
        # Ensure credentials / token are available before attempting download
        token = self._token_value()

        last_error: Optional[Exception] = None
        for attempt in range(self.settings.stac_max_retries):
            try:
                headers = {
                    "User-Agent": "GlacierGuard-AI/phase8",
                    "Authorization": f"Bearer {token}",
                }
                request = urllib.request.Request(href, headers=headers)
                with urllib.request.urlopen(request, timeout=self.settings.asset_timeout_seconds) as response:
                    content = response.read()

                if not content:
                    raise AssetAccessError(f"Asset {key!r} download was empty")

                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                self._validate_raster(destination, key)

                return AccessedAsset(
                    key=key,
                    path=destination,
                    href_host=urllib.parse.urlparse(href).netloc,
                    byte_count=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                )
            except AssetAccessError:
                destination.unlink(missing_ok=True)
                raise
            except (urllib.error.URLError, OSError) as exc:
                last_error = exc
                destination.unlink(missing_ok=True)
                if attempt + 1 < self.settings.stac_max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    self._token = None  # Force fresh token on next attempt

        raise AssetAccessError(
            f"Unable to access asset {key!r} via HTTPS: {last_error}"
        ) from last_error

    def fetch(self, product_id: str, key: str, asset: Dict[str, Any]) -> AccessedAsset:
        """Download or retrieve one declared STAC asset with cache validation.

        Parameters
        ----------
        product_id : str
            Parent satellite product ID.
        key : str
            Asset identifier key (e.g. 'B03_20m', 'VV').
        asset : Dict[str, Any]
            STAC asset metadata dictionary containing 'href'.

        Returns
        -------
        AccessedAsset
            Accessed and verified asset reference.
        """
        href = asset.get("href") if isinstance(asset, dict) else None
        if not href or not isinstance(href, str):
            raise AssetAccessError(f"Asset {key!r} has no usable href")

        parsed = urllib.parse.urlparse(href)
        scheme = parsed.scheme.lower()

        if scheme not in ("file", "https", "http", "s3"):
            raise AssetAccessError(
                f"Unsupported asset URL scheme {scheme!r} in {href}. "
                f"GlacierGuard-AI does not silently rewrite unsupported URLs."
            )

        # Check existing cached destination
        suffix = Path(parsed.path).suffix or ".tif"
        safe_product = hashlib.sha256(product_id.encode("utf-8")).hexdigest()[:16]
        destination = (
            self.settings.processing_cache_dir / safe_product / f"{key}{suffix}"
        )

        if destination.is_file() and destination.stat().st_size > 0:
            try:
                self._validate_raster(destination, key)
                content = destination.read_bytes()
                return AccessedAsset(
                    key=key,
                    path=destination,
                    href_host=parsed.netloc or "cached",
                    byte_count=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                )
            except AssetAccessError:
                destination.unlink(missing_ok=True)

        if scheme == "file":
            return self._fetch_file_url(href, destination, key)
        elif scheme == "s3":
            return self._fetch_s3_url(href, destination, key)
        else:
            return self._fetch_http_url(href, destination, key)
