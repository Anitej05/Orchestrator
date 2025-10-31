# auth.py
# Minimal Clerk JWT verification for FastAPI using python-jose

import os
from typing import Optional, Dict, Any, Tuple
from fastapi import Request,HTTPException, status
from jose import jwt, JWTError
import requests
import httpx
import logging

logger = logging.getLogger(__name__)

# Simple in-process cache for JWKS
_JWKS_CACHE: Dict[str, Dict[str, Any]] = {}

# Expected Clerk settings (from frontend Clerk):
# - CLERK_JWT_AUDIENCE (optional)
# - CLERK_JWT_ISSUER (e.g., https://clerk.YOUR_DOMAIN)
# - CLERK_JWT_PUBLIC_KEY (PEM) or we can rely on JWKS in production

"""
Supported env vars:
- CLERK_JWT_PUBLIC_KEY: PEM public key (optional)
- CLERK_JWKS_URL: JWKS endpoint (preferred; see Clerk dashboard screenshot)
- CLERK_JWT_ISSUER: expected issuer
- CLERK_JWT_AUDIENCE: expected audience (optional)
"""
CLERK_JWT_PUBLIC_KEY = os.getenv("CLERK_JWT_PUBLIC_KEY")
CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")
CLERK_JWT_ISSUER = os.getenv("CLERK_JWT_ISSUER")
CLERK_JWT_AUDIENCE = os.getenv("CLERK_JWT_AUDIENCE")


def _extract_bearer_token(authorization_header: Optional[str]) -> str:
	if not authorization_header:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
	parts = authorization_header.split()
	if len(parts) != 2 or parts[0].lower() != "bearer":
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header format")
	return parts[1]


def verify_clerk_token(authorization_header: Optional[str]) -> Dict[str, Any]:
    """
    Verify the Clerk JWT and return claims.
    Prefers JWKS verification if CLERK_JWKS_URL is set; falls back to PEM.
    """
    token = _extract_bearer_token(authorization_header)

    # Try JWKS path first
    if CLERK_JWKS_URL:
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if not kid:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing kid in token header")

            jwks = _JWKS_CACHE.get(CLERK_JWKS_URL)
            if not jwks:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(CLERK_JWKS_URL)
                    resp.raise_for_status()
                    jwks = resp.json()
                _JWKS_CACHE[CLERK_JWKS_URL] = jwks

            keys = jwks.get("keys", [])
            public_key = None
            algorithm = None
            for key in keys:
                if key.get("kid") == kid:
                    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
                    algorithm = key.get("alg") or "RS256"
                    break
            if not public_key:
                # refresh once in case of rotation
                _JWKS_CACHE.pop(CLERK_JWKS_URL, None)
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(CLERK_JWKS_URL)
                    resp.raise_for_status()
                    jwks = resp.json()
                _JWKS_CACHE[CLERK_JWKS_URL] = jwks
                for key in jwks.get("keys", []):
                    if key.get("kid") == kid:
                        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
                        algorithm = key.get("alg") or "RS256"
                        break
            if not public_key:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Signing key not found")

            options = {"verify_aud": bool(CLERK_JWT_AUDIENCE)}
            claims = jwt.decode(
                token,
                public_key,
                algorithms=[algorithm],
                issuer=CLERK_JWT_ISSUER if CLERK_JWT_ISSUER else None,
                audience=CLERK_JWT_AUDIENCE if CLERK_JWT_AUDIENCE else None,
                options=options,
            )
            return claims
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    # Fallback to PEM if provided
    if CLERK_JWT_PUBLIC_KEY:
        try:
            options = {"verify_aud": bool(CLERK_JWT_AUDIENCE)}
            claims = jwt.decode(
                token,
                CLERK_JWT_PUBLIC_KEY,
                algorithms=["RS256"],
                issuer=CLERK_JWT_ISSUER if CLERK_JWT_ISSUER else None,
                audience=CLERK_JWT_AUDIENCE if CLERK_JWT_AUDIENCE else None,
                options=options,
            )
            return claims
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server missing CLERK_JWKS_URL or CLERK_JWT_PUBLIC_KEY")


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
