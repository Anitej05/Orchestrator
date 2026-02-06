# auth.py
# Minimal Clerk JWT verification for FastAPI using python-jose

import os
import time
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status
from jose import jwt, JWTError
from jose.utils import base64url_decode
import requests
import logging

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

# --- JWKS Caching ---
_JWKS_CACHE: Dict[str, Any] = {}
_JWKS_CACHE_TTL = 300  # 5 minutes

"""
Supported env vars:
- CLERK_JWT_PUBLIC_KEY: PEM public key (optional)
- CLERK_JWKS_URL: JWKS endpoint (preferred; see Clerk dashboard screenshot)
- CLERK_JWT_ISSUER: expected issuer
- CLERK_JWT_AUDIENCE: expected audience (optional)
"""
CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")
CLERK_JWT_ISSUER = os.getenv("CLERK_JWT_ISSUER")
CLERK_JWT_AUDIENCE = os.getenv("CLERK_JWT_AUDIENCE")


# =============================================================================
# HELPER FUNCTIONS (Extracted to avoid duplication)
# =============================================================================

def _extract_bearer_token(authorization_header: Optional[str]) -> str:
    """Extract JWT from Bearer token header."""
    if not authorization_header:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    parts = authorization_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header format")
    return parts[1]


def _fetch_jwks() -> Dict[str, Any]:
    """
    Fetch JWKS from Clerk with caching.
    Returns cached JWKS if available and not expired.
    """
    global _JWKS_CACHE
    
    now = time.time()
    if _JWKS_CACHE.get("data") and _JWKS_CACHE.get("expires_at", 0) > now:
        return _JWKS_CACHE["data"]
    
    # Fetch fresh JWKS
    jwks_url = os.getenv("CLERK_JWKS_URL")
    if not jwks_url:
        raise HTTPException(status_code=500, detail="CLERK_JWKS_URL not configured")
    
    try:
        response = requests.get(jwks_url, timeout=10)
        response.raise_for_status()
        jwks = response.json()
        
        # Cache it
        _JWKS_CACHE = {
            "data": jwks,
            "expires_at": now + _JWKS_CACHE_TTL
        }
        logger.debug("JWKS cache refreshed")
        return jwks
    except Exception as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        # If we have stale cache, use it as fallback
        if _JWKS_CACHE.get("data"):
            logger.warning("Using stale JWKS cache due to fetch failure")
            return _JWKS_CACHE["data"]
        raise HTTPException(status_code=500, detail="Failed to fetch JWKS")


def _jwk_to_pem(jwk: Dict[str, Any]) -> bytes:
    """Convert JWK to PEM format. Only supports RSA keys."""
    if jwk.get("kty") != "RSA":
        raise ValueError("Only RSA keys are supported")
    
    n = int.from_bytes(base64url_decode(jwk["n"].encode()), "big")
    e = int.from_bytes(base64url_decode(jwk["e"].encode()), "big")
    
    pubkey = rsa.RSAPublicNumbers(e, n).public_key()
    return pubkey.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )


def _get_public_key_for_token(token: str) -> bytes:
    """Get the correct public key for a JWT from JWKS."""
    jwks = _fetch_jwks()
    
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token header: {e}")
    
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return _jwk_to_pem(key)
    
    logger.error(f"No matching key found for kid: {kid}")
    raise HTTPException(status_code=401, detail="Invalid token: No matching key")


def _decode_and_verify_token(token: str) -> Dict[str, Any]:
    """Decode and verify a JWT token."""
    public_key = _get_public_key_for_token(token)
    
    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=CLERK_JWT_AUDIENCE,
            issuer=CLERK_JWT_ISSUER
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.error("JWT expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError as e:
        logger.error(f"JWT error: {e}")
        raise HTTPException(status_code=401, detail=f"JWT error: {str(e)}")


# =============================================================================
# PUBLIC API
# =============================================================================

def verify_clerk_token(authorization_header: Optional[str]) -> Dict[str, Any]:
    """
    Verify the Clerk JWT and return claims.
    Uses cached JWKS for performance.
    """
    token = _extract_bearer_token(authorization_header)
    
    try:
        payload = _decode_and_verify_token(token)
        logger.info(f"User authenticated via verify_clerk_token: user_id={payload.get('sub')}")
        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_user_from_request(request: Request) -> Dict[str, Any]:
    """
    Extract and verify user from request Authorization header.
    Returns the JWT payload with user claims.
    """
    auth_header = request.headers.get("Authorization")
    logger.debug(f"Authorization header present: {bool(auth_header)}")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error("Missing or invalid Authorization header")
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = auth_header.split(" ")[1]
    
    try:
        payload = _decode_and_verify_token(token)
        logger.info(f"User authenticated: user_id={payload.get('sub')}, email={payload.get('email')}")
        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_current_user_id(request: Request) -> str:
    """
    Extract user ID from JWT token in request.
    Used by credential management endpoints.
    """
    try:
        payload = get_user_from_request(request)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token"
            )
        
        return user_id
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extract user ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to authenticate user"
        )

    """
    Verify the Clerk JWT and return claims.
    Uses the same working JWKS verification as get_user_from_request.
    """
    token = _extract_bearer_token(authorization_header)
    
    try:
        # Fetch JWKS
        jwks_response = requests.get(os.getenv("CLERK_JWKS_URL"))
        jwks = jwks_response.json()
        public_key = None
        from jose.utils import base64url_decode
        
        def jwk_to_pem(jwk):
            # Only supports RSA keys
            if jwk["kty"] != "RSA":
                raise ValueError("Only RSA keys are supported")
            n = int.from_bytes(base64url_decode(jwk["n"].encode()), "big")
            e = int.from_bytes(base64url_decode(jwk["e"].encode()), "big")
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization
            pubkey = rsa.RSAPublicNumbers(e, n).public_key()
            return pubkey.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        
        for key in jwks["keys"]:
            if key["kid"] == jwt.get_unverified_header(token)["kid"]:
                public_key = jwk_to_pem(key)
                break
        
        if not public_key:
            logger.error("Invalid token: No matching key")
            raise HTTPException(status_code=401, detail="Invalid token: No matching key")
        
        # Decode and verify
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=os.getenv("CLERK_JWT_AUDIENCE"),
            issuer=os.getenv("CLERK_JWT_ISSUER")
        )
        logger.info(f"User authenticated via verify_clerk_token: user_id={payload.get('sub')}")
        return payload
        
    except jwt.ExpiredSignatureError:
        logger.error("JWT expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError as e:
        logger.error(f"JWT error: {e}")
        raise HTTPException(status_code=401, detail=f"JWT error: {str(e)}")
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_user_from_request(request: Request):
    auth_header = request.headers.get("Authorization")
    logger.info(f"Authorization header present: {bool(auth_header)}")
    if auth_header:
        logger.info(f"Authorization header value: {auth_header[:50]}...")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error("Missing or invalid Authorization header")
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = auth_header.split(" ")[1]
    try:
        # Fetch JWKS
        jwks_response = requests.get(os.getenv("CLERK_JWKS_URL"))
        jwks = jwks_response.json()
        public_key = None
        from jose.utils import base64url_decode
        def jwk_to_pem(jwk):
            # Only supports RSA keys
            if jwk["kty"] != "RSA":
                raise ValueError("Only RSA keys are supported")
            n = int.from_bytes(base64url_decode(jwk["n"].encode()), "big")
            e = int.from_bytes(base64url_decode(jwk["e"].encode()), "big")
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization
            pubkey = rsa.RSAPublicNumbers(e, n).public_key()
            return pubkey.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        for key in jwks["keys"]:
            if key["kid"] == jwt.get_unverified_header(token)["kid"]:
                public_key = jwk_to_pem(key)
                break
        if not public_key:
            logger.error("Invalid token: No matching key")
            raise HTTPException(status_code=401, detail="Invalid token: No matching key")
        # Decode and verify
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=os.getenv("CLERK_JWT_AUDIENCE"),
            issuer=os.getenv("CLERK_JWT_ISSUER")
        )
        logger.info(f"Full JWT payload: {payload}")
        logger.info(f"User authenticated successfully: user_id={payload.get('sub')}, email={payload.get('email')}")
        return payload
    except jwt.ExpiredSignatureError:
        logger.error("JWT expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError as e:
        # python-jose does not have InvalidAudienceError/InvalidIssuerError, so handle generically
        logger.error(f"JWT error: {e}")
        raise HTTPException(status_code=401, detail=f"JWT error: {str(e)}")


def get_current_user_id(request: Request) -> str:
    """
    Extract user ID from JWT token in request.
    Used by credential management endpoints.
    """
    try:
        payload = get_user_from_request(request)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token"
            )
        
        return user_id
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extract user ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to authenticate user"
        )
