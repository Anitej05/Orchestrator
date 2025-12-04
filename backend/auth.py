# auth.py
# Minimal Clerk JWT verification for FastAPI using python-jose

import os
from typing import Optional, Dict, Any, Tuple
from fastapi import Request,HTTPException, status
from jose import jwt, JWTError
from jose.backends import RSAKey
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
