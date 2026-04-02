import os
import time
from typing import Optional

import requests
from jose import jwt

REGION = os.environ.get("REGION", "ap-northeast-1")
USER_POOL_ID = os.environ.get("USER_POOL_ID", "")
CLIENT_ID = os.environ.get("CLIENT_ID", "")

# Cache JWKS at module level to avoid fetching on every request.
# Keys are rotated infrequently; 10-minute TTL is a safe balance.
_JWKS_CACHE: Optional[list] = None
_JWKS_CACHE_EXPIRY: float = 0.0
_JWKS_CACHE_TTL = 600  # seconds


def _get_jwks() -> list:
    global _JWKS_CACHE, _JWKS_CACHE_EXPIRY
    if _JWKS_CACHE is not None and time.monotonic() < _JWKS_CACHE_EXPIRY:
        return _JWKS_CACHE
    url = f"https://cognito-idp.{REGION}.amazonaws.com/{USER_POOL_ID}/.well-known/jwks.json"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    _JWKS_CACHE = response.json()["keys"]
    _JWKS_CACHE_EXPIRY = time.monotonic() + _JWKS_CACHE_TTL
    return _JWKS_CACHE


def verify_token(token: str) -> dict:
    # Verify JWT token
    keys = _get_jwks()
    header = jwt.get_unverified_header(token)
    matching_keys = [k for k in keys if k["kid"] == header["kid"]]
    if not matching_keys:
        raise ValueError(f"No matching key found for kid: {header.get('kid')}")
    key = matching_keys[0]
    # The JWT returned from the Identity Provider may contain an at_hash
    # jose jwt.decode verifies id_token with access_token by default if it contains at_hash
    # See : https://github.com/mpdavis/python-jose/blob/4b0701b46a8d00988afcc5168c2b3a1fd60d15d8/jose/jwt.py#L59
    # Since we are not using an access token in the app, skipping the verification of the at_hash.
    # so we will disable the verify_at_hash check.
    decoded = jwt.decode(
        token,
        key,
        algorithms=["RS256"],
        options={"verify_at_hash": False},
        audience=CLIENT_ID,
    )
    return decoded
