"""
Unit tests for backend/auth.py and backend/utils/file_server.py

Coverage:
 auth.py
  - _extract_bearer_token: missing/empty header, wrong scheme, single word, three parts, valid, case-insensitive
  - _fetch_jwks: cache hit, expired cache fetches fresh, stale fallback on failure, no cache+failure→500, missing URL→500
  - _jwk_to_pem: valid RSA round-trip, non-RSA kty raises ValueError
  - _get_public_key_for_token: matching kid, no match→401, invalid token header→401
  - _decode_and_verify_token: valid token, expired→401, wrong key signature→401, wrong audience→401
  - verify_clerk_token: full-flow success, missing header, expired token
  - get_user_from_request: dev-mode (no header / bearer / sub extraction), prod (valid, missing header, non-bearer, expired)
  - get_current_user_id: returns sub, missing sub→401, auth exception propagates

 file_server.py
  - is_safe_path: double-dot, nested double-dot, tilde, double-slash, double-backslash, absolute unix, valid relative, simple filename
  - resolve_and_validate_path: within storage, escape via .., resolved path includes filename, subdirectory
  - find_file_in_storage: not found, path-style direct lookup, single match, newest selected from multiple
  - get_mime_type: pdf, xlsx, png, csv, json, unknown→octet-stream
  - should_inline_preview: pdf/text/image/csv/json inline; docx/xlsx/octet-stream not inline
"""

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from jose import jwt

import tempfile

import backend.auth as auth_module
from backend.auth import (
    _decode_and_verify_token,
    _extract_bearer_token,
    _fetch_jwks,
    _get_public_key_for_token,
    _jwk_to_pem,
    get_current_user_id,
    get_user_from_request,
    verify_clerk_token,
)
from backend.utils.file_server import (
    STORAGE_BASE,
    find_file_in_storage,
    get_mime_type,
    is_safe_path,
    resolve_and_validate_path,
    should_inline_preview,
)


# ── Key generation helpers ────────────────────────────────────────────────────

def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _int_to_b64url(n: int) -> str:
    byte_length = (n.bit_length() + 7) // 8
    return _b64url(n.to_bytes(byte_length, "big"))


@pytest.fixture(scope="session")
def rsa_key_pair():
    """RSA-2048 key pair + matching JWK generated once per session."""
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    public_key = private_key.public_key()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_numbers = public_key.public_numbers()
    jwk = {
        "kty": "RSA",
        "kid": "test-kid-1",
        "alg": "RS256",
        "use": "sig",
        "n": _int_to_b64url(pub_numbers.n),
        "e": _int_to_b64url(pub_numbers.e),
    }
    return {
        "private_key": private_key,
        "public_key": public_key,
        "private_pem": private_pem,
        "jwk": jwk,
        "jwks": {"keys": [jwk]},
        "kid": "test-kid-1",
    }


def _make_token(private_pem: bytes, kid: str, claims: Dict[str, Any]) -> str:
    return jwt.encode(claims, private_pem, algorithm="RS256", headers={"kid": kid})


def _make_request(auth_header=None):
    """Mock FastAPI Request with optional Authorization header."""
    req = MagicMock()
    headers: Dict[str, str] = {}
    if auth_header is not None:
        headers["Authorization"] = auth_header
    req.headers.get = lambda key, default=None: headers.get(key, default)
    return req


@pytest.fixture(autouse=True)
def reset_jwks_cache():
    """Wipe the module-level JWKS cache before and after every test."""
    auth_module._JWKS_CACHE.clear()
    yield
    auth_module._JWKS_CACHE.clear()


def _env_without_jwks_url() -> Dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "CLERK_JWKS_URL"}


# ── _extract_bearer_token ─────────────────────────────────────────────────────

class TestExtractBearerToken:
    def test_none_header_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            _extract_bearer_token(None)
        assert exc.value.status_code == 401
        assert "Missing" in exc.value.detail

    def test_empty_string_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            _extract_bearer_token("")
        assert exc.value.status_code == 401

    def test_wrong_scheme_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            _extract_bearer_token("Basic dXNlcjpwYXNz")
        assert exc.value.status_code == 401
        assert "Invalid" in exc.value.detail

    def test_bearer_only_no_token_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            _extract_bearer_token("Bearer")
        assert exc.value.status_code == 401

    def test_three_parts_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            _extract_bearer_token("Bearer tok1 extra")
        assert exc.value.status_code == 401

    def test_valid_header_returns_token(self):
        assert _extract_bearer_token("Bearer mytoken123") == "mytoken123"

    def test_case_insensitive_scheme(self):
        assert _extract_bearer_token("bearer mytoken123") == "mytoken123"
        assert _extract_bearer_token("BEARER mytoken123") == "mytoken123"


# ── _fetch_jwks ───────────────────────────────────────────────────────────────

class TestFetchJwks:
    def test_fresh_cache_returned_without_http_call(self):
        fake_jwks = {"keys": [{"kid": "cached"}]}
        auth_module._JWKS_CACHE = {
            "data": fake_jwks,
            "expires_at": time.time() + 3600,
        }
        with patch("backend.auth.requests.get") as mock_get:
            result = _fetch_jwks()
        mock_get.assert_not_called()
        assert result == fake_jwks

    def test_expired_cache_triggers_fresh_fetch(self):
        old_jwks = {"keys": [{"kid": "old"}]}
        new_jwks = {"keys": [{"kid": "new"}]}
        auth_module._JWKS_CACHE = {"data": old_jwks, "expires_at": time.time() - 1}
        mock_resp = MagicMock()
        mock_resp.json.return_value = new_jwks
        with patch.dict(os.environ, {"CLERK_JWKS_URL": "https://example.com/jwks.json"}):
            with patch("backend.auth.requests.get", return_value=mock_resp):
                result = _fetch_jwks()
        assert result == new_jwks

    def test_fetch_failure_falls_back_to_stale_cache(self):
        stale_jwks = {"keys": [{"kid": "stale"}]}
        auth_module._JWKS_CACHE = {"data": stale_jwks, "expires_at": time.time() - 1}
        with patch.dict(os.environ, {"CLERK_JWKS_URL": "https://example.com/jwks.json"}):
            with patch("backend.auth.requests.get", side_effect=ConnectionError("down")):
                result = _fetch_jwks()
        assert result == stale_jwks

    def test_no_cache_and_fetch_failure_raises_500(self):
        with patch.dict(os.environ, {"CLERK_JWKS_URL": "https://example.com/jwks.json"}):
            with patch("backend.auth.requests.get", side_effect=ConnectionError("down")):
                with pytest.raises(HTTPException) as exc:
                    _fetch_jwks()
        assert exc.value.status_code == 500

    def test_missing_jwks_url_raises_500_with_message(self):
        with patch.dict(os.environ, _env_without_jwks_url(), clear=True):
            with pytest.raises(HTTPException) as exc:
                _fetch_jwks()
        assert exc.value.status_code == 500
        assert "CLERK_JWKS_URL" in exc.value.detail

    def test_successful_fetch_updates_cache(self):
        jwks = {"keys": [{"kid": "fresh"}]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = jwks
        with patch.dict(os.environ, {"CLERK_JWKS_URL": "https://example.com/jwks.json"}):
            with patch("backend.auth.requests.get", return_value=mock_resp):
                _fetch_jwks()
        assert auth_module._JWKS_CACHE["data"] == jwks
        assert auth_module._JWKS_CACHE["expires_at"] > time.time()


# ── _jwk_to_pem ───────────────────────────────────────────────────────────────

class TestJwkToPem:
    def test_rsa_jwk_produces_pem_bytes(self, rsa_key_pair):
        pem = _jwk_to_pem(rsa_key_pair["jwk"])
        assert isinstance(pem, bytes)
        assert b"BEGIN PUBLIC KEY" in pem

    def test_non_rsa_kty_raises_value_error(self):
        ec_jwk = {"kty": "EC", "kid": "ec-1", "crv": "P-256", "x": "abc", "y": "def"}
        with pytest.raises(ValueError, match="RSA"):
            _jwk_to_pem(ec_jwk)

    def test_round_trip_matches_original_public_key(self, rsa_key_pair):
        expected = rsa_key_pair["public_key"].public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        assert _jwk_to_pem(rsa_key_pair["jwk"]) == expected


# ── _get_public_key_for_token ─────────────────────────────────────────────────

class TestGetPublicKeyForToken:
    def test_matching_kid_returns_pem(self, rsa_key_pair):
        token = _make_token(
            rsa_key_pair["private_pem"],
            rsa_key_pair["kid"],
            {"sub": "u1", "exp": int(time.time()) + 3600},
        )
        with patch("backend.auth._fetch_jwks", return_value=rsa_key_pair["jwks"]):
            pem = _get_public_key_for_token(token)
        assert b"BEGIN PUBLIC KEY" in pem

    def test_no_matching_kid_raises_401(self, rsa_key_pair):
        token = _make_token(
            rsa_key_pair["private_pem"],
            "different-kid",
            {"sub": "u1", "exp": int(time.time()) + 3600},
        )
        with patch("backend.auth._fetch_jwks", return_value=rsa_key_pair["jwks"]):
            with pytest.raises(HTTPException) as exc:
                _get_public_key_for_token(token)
        assert exc.value.status_code == 401
        assert "No matching key" in exc.value.detail

    def test_invalid_token_raises_401(self):
        with patch("backend.auth._fetch_jwks", return_value={"keys": []}):
            with pytest.raises(HTTPException) as exc:
                _get_public_key_for_token("not.a.jwt")
        assert exc.value.status_code == 401


# ── _decode_and_verify_token ──────────────────────────────────────────────────

class TestDecodeAndVerifyToken:
    def _with_nulled_audience_issuer(self):
        """Context stacking helper: patches AUDIENCE and ISSUER to None."""
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(auth_module, "CLERK_JWT_AUDIENCE", None))
        stack.enter_context(patch.object(auth_module, "CLERK_JWT_ISSUER", None))
        return stack

    def test_valid_token_returns_payload(self, rsa_key_pair):
        claims = {"sub": "user_abc", "exp": int(time.time()) + 3600}
        token = _make_token(rsa_key_pair["private_pem"], rsa_key_pair["kid"], claims)
        with patch("backend.auth._fetch_jwks", return_value=rsa_key_pair["jwks"]):
            with self._with_nulled_audience_issuer():
                payload = _decode_and_verify_token(token)
        assert payload["sub"] == "user_abc"

    def test_expired_token_raises_401(self, rsa_key_pair):
        claims = {"sub": "user_abc", "exp": int(time.time()) - 100}
        token = _make_token(rsa_key_pair["private_pem"], rsa_key_pair["kid"], claims)
        with patch("backend.auth._fetch_jwks", return_value=rsa_key_pair["jwks"]):
            with self._with_nulled_audience_issuer():
                with pytest.raises(HTTPException) as exc:
                    _decode_and_verify_token(token)
        assert exc.value.status_code == 401
        assert "expired" in exc.value.detail.lower()

    def test_wrong_signing_key_raises_401(self, rsa_key_pair):
        """Token signed with a different private key fails verification."""
        other_private = rsa.generate_private_key(65537, 2048, default_backend())
        other_pem = other_private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        claims = {"sub": "user_abc", "exp": int(time.time()) + 3600}
        # Signed with other_pem but JWKS holds the original public key
        token = _make_token(other_pem, rsa_key_pair["kid"], claims)
        with patch("backend.auth._fetch_jwks", return_value=rsa_key_pair["jwks"]):
            with self._with_nulled_audience_issuer():
                with pytest.raises(HTTPException) as exc:
                    _decode_and_verify_token(token)
        assert exc.value.status_code == 401

    def test_wrong_audience_raises_401(self, rsa_key_pair):
        claims = {
            "sub": "user_abc",
            "aud": "wrong-audience",
            "exp": int(time.time()) + 3600,
        }
        token = _make_token(rsa_key_pair["private_pem"], rsa_key_pair["kid"], claims)
        with patch("backend.auth._fetch_jwks", return_value=rsa_key_pair["jwks"]):
            with patch.object(auth_module, "CLERK_JWT_AUDIENCE", "expected-audience"):
                with patch.object(auth_module, "CLERK_JWT_ISSUER", None):
                    with pytest.raises(HTTPException) as exc:
                        _decode_and_verify_token(token)
        assert exc.value.status_code == 401


# ── verify_clerk_token ────────────────────────────────────────────────────────

class TestVerifyClerkToken:
    def test_valid_header_returns_claims(self, rsa_key_pair):
        claims = {"sub": "u1", "exp": int(time.time()) + 3600}
        token = _make_token(rsa_key_pair["private_pem"], rsa_key_pair["kid"], claims)
        with patch("backend.auth._fetch_jwks", return_value=rsa_key_pair["jwks"]):
            with patch.object(auth_module, "CLERK_JWT_AUDIENCE", None):
                with patch.object(auth_module, "CLERK_JWT_ISSUER", None):
                    result = verify_clerk_token(f"Bearer {token}")
        assert result["sub"] == "u1"

    def test_missing_header_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            verify_clerk_token(None)
        assert exc.value.status_code == 401

    def test_expired_token_raises_401(self, rsa_key_pair):
        claims = {"sub": "u1", "exp": int(time.time()) - 60}
        token = _make_token(rsa_key_pair["private_pem"], rsa_key_pair["kid"], claims)
        with patch("backend.auth._fetch_jwks", return_value=rsa_key_pair["jwks"]):
            with patch.object(auth_module, "CLERK_JWT_AUDIENCE", None):
                with patch.object(auth_module, "CLERK_JWT_ISSUER", None):
                    with pytest.raises(HTTPException) as exc:
                        verify_clerk_token(f"Bearer {token}")
        assert exc.value.status_code == 401

    def test_malformed_token_raises_401(self):
        with patch.dict(os.environ, {"CLERK_JWKS_URL": "https://example.com/jwks"}):
            with patch("backend.auth._fetch_jwks", return_value={"keys": []}):
                with pytest.raises(HTTPException) as exc:
                    verify_clerk_token("Bearer thisisnotajwt")
        assert exc.value.status_code == 401


# ── get_user_from_request ─────────────────────────────────────────────────────

class TestGetUserFromRequest:
    # ── Dev mode ──────────────────────────────────────────────────────────────

    def test_dev_mode_no_header_returns_dev_user(self):
        req = _make_request(None)
        with patch.dict(os.environ, _env_without_jwks_url(), clear=True):
            result = get_user_from_request(req)
        assert result["sub"] == "dev-user"
        assert result["dev_mode"] is True

    def test_dev_mode_malformed_bearer_returns_dev_user(self):
        """Unparseable token in dev mode → fallback to 'dev-user'."""
        req = _make_request("Bearer notajwtatall")
        with patch.dict(os.environ, _env_without_jwks_url(), clear=True):
            result = get_user_from_request(req)
        assert result["sub"] == "dev-user"
        assert result["dev_mode"] is True

    def test_dev_mode_reads_sub_from_unverified_claims(self, rsa_key_pair):
        """Dev mode extracts sub from JWT without verifying signature."""
        claims = {"sub": "dev-user-123", "exp": int(time.time()) + 3600}
        token = _make_token(rsa_key_pair["private_pem"], rsa_key_pair["kid"], claims)
        req = _make_request(f"Bearer {token}")
        with patch.dict(os.environ, _env_without_jwks_url(), clear=True):
            result = get_user_from_request(req)
        assert result["sub"] == "dev-user-123"
        assert result["dev_mode"] is True

    def test_dev_mode_returns_user_id_equals_sub(self):
        """user_id field mirrors sub in dev mode response."""
        req = _make_request(None)
        with patch.dict(os.environ, _env_without_jwks_url(), clear=True):
            result = get_user_from_request(req)
        assert result["user_id"] == result["sub"]

    # ── Production mode ───────────────────────────────────────────────────────

    def test_prod_mode_valid_token_returns_payload(self):
        payload = {"sub": "prod-user-1", "exp": int(time.time()) + 3600}
        req = _make_request("Bearer sometoken")
        with patch.dict(os.environ, {"CLERK_JWKS_URL": "https://example.com/jwks"}):
            with patch("backend.auth._decode_and_verify_token", return_value=payload):
                result = get_user_from_request(req)
        assert result["sub"] == "prod-user-1"

    def test_prod_mode_missing_header_raises_401(self):
        req = _make_request(None)
        with patch.dict(os.environ, {"CLERK_JWKS_URL": "https://example.com/jwks"}):
            with pytest.raises(HTTPException) as exc:
                get_user_from_request(req)
        assert exc.value.status_code == 401

    def test_prod_mode_non_bearer_header_raises_401(self):
        req = _make_request("Basic dXNlcjpwYXNz")
        with patch.dict(os.environ, {"CLERK_JWKS_URL": "https://example.com/jwks"}):
            with pytest.raises(HTTPException) as exc:
                get_user_from_request(req)
        assert exc.value.status_code == 401

    def test_prod_mode_expired_token_raises_401(self, rsa_key_pair):
        claims = {"sub": "u1", "exp": int(time.time()) - 100}
        token = _make_token(rsa_key_pair["private_pem"], rsa_key_pair["kid"], claims)
        req = _make_request(f"Bearer {token}")
        with patch.dict(os.environ, {"CLERK_JWKS_URL": "https://example.com/jwks"}):
            with patch("backend.auth._fetch_jwks", return_value=rsa_key_pair["jwks"]):
                with patch.object(auth_module, "CLERK_JWT_AUDIENCE", None):
                    with patch.object(auth_module, "CLERK_JWT_ISSUER", None):
                        with pytest.raises(HTTPException) as exc:
                            get_user_from_request(req)
        assert exc.value.status_code == 401


# ── get_current_user_id ───────────────────────────────────────────────────────

class TestGetCurrentUserId:
    def test_returns_sub_from_token(self):
        req = _make_request()
        with patch("backend.auth.get_user_from_request", return_value={"sub": "user-xyz"}):
            assert get_current_user_id(req) == "user-xyz"

    def test_missing_sub_raises_401(self):
        req = _make_request()
        with patch("backend.auth.get_user_from_request", return_value={"email": "x@x.com"}):
            with pytest.raises(HTTPException) as exc:
                get_current_user_id(req)
        assert exc.value.status_code == 401
        assert "User ID not found" in exc.value.detail

    def test_auth_exception_propagates(self):
        req = _make_request()
        with patch(
            "backend.auth.get_user_from_request",
            side_effect=HTTPException(401, "Unauthorized"),
        ):
            with pytest.raises(HTTPException) as exc:
                get_current_user_id(req)
        assert exc.value.status_code == 401

    def test_unexpected_exception_raises_401(self):
        req = _make_request()
        with patch(
            "backend.auth.get_user_from_request",
            side_effect=RuntimeError("unexpected"),
        ):
            with pytest.raises(HTTPException) as exc:
                get_current_user_id(req)
        assert exc.value.status_code == 401


# ── is_safe_path ──────────────────────────────────────────────────────────────

class TestIsSafePath:
    def test_double_dot_blocked(self):
        assert is_safe_path("../etc/passwd") is False

    def test_nested_double_dot_blocked(self):
        assert is_safe_path("documents/../../etc/passwd") is False

    def test_tilde_blocked(self):
        assert is_safe_path("~/secrets") is False

    def test_tilde_in_middle_blocked(self):
        assert is_safe_path("documents/~/hidden") is False

    def test_double_slash_blocked(self):
        assert is_safe_path("documents//secret") is False

    def test_double_backslash_blocked(self):
        # Python string "documents\\\\secret" contains literal \\
        assert is_safe_path("documents\\\\secret") is False

    def test_absolute_windows_path_blocked(self):
        # Windows-style absolute path (drive letter + backslash)
        assert is_safe_path("C:\\Windows\\System32\\file.txt") is False

    @pytest.mark.skipif(os.name != "posix", reason="Unix-only: posix isabs check")
    def test_absolute_unix_path_blocked(self):
        assert is_safe_path("/etc/passwd") is False

    def test_valid_relative_path_allowed(self):
        assert is_safe_path("documents/report.pdf") is True

    def test_simple_filename_allowed(self):
        assert is_safe_path("report.pdf") is True

    def test_nested_relative_path_allowed(self):
        assert is_safe_path("spreadsheets/q1/data.xlsx") is True


# ── resolve_and_validate_path ─────────────────────────────────────────────────

class TestResolveAndValidatePath:
    def test_valid_relative_path_stays_within_storage(self):
        result = resolve_and_validate_path("documents/test.pdf")
        assert result is not None
        assert str(result).startswith(str(STORAGE_BASE))

    def test_path_escape_via_dotdot_returns_none(self):
        """Even without is_safe_path, the resolve+startswith check blocks escape."""
        result = resolve_and_validate_path("../../etc/passwd")
        assert result is None

    def test_resolved_path_includes_final_filename(self):
        result = resolve_and_validate_path("images/photo.png")
        assert result is not None
        assert result.name == "photo.png"

    def test_deep_subdirectory_path_within_storage(self):
        result = resolve_and_validate_path("spreadsheets/2024/data.xlsx")
        assert result is not None
        assert "spreadsheets" in str(result)

    def test_deeply_nested_dotdot_escape_blocked(self):
        result = resolve_and_validate_path("a/b/c/../../../../../../../etc/passwd")
        assert result is None


# ── get_mime_type ─────────────────────────────────────────────────────────────

class TestGetMimeType:
    def test_pdf(self):
        assert get_mime_type(Path("report.pdf")) == "application/pdf"

    def test_xlsx(self):
        assert "spreadsheetml" in get_mime_type(Path("data.xlsx"))

    def test_docx(self):
        assert "wordprocessingml" in get_mime_type(Path("doc.docx"))

    def test_png(self):
        assert get_mime_type(Path("image.png")) == "image/png"

    def test_jpeg(self):
        mime = get_mime_type(Path("photo.jpg"))
        assert mime == "image/jpeg"

    def test_csv(self):
        # Windows mimetypes DB maps .csv to application/vnd.ms-excel; accept both
        mime = get_mime_type(Path("data.csv"))
        assert mime in {"text/csv", "application/vnd.ms-excel"}

    def test_json(self):
        assert get_mime_type(Path("data.json")) == "application/json"

    def test_txt(self):
        assert get_mime_type(Path("notes.txt")) == "text/plain"

    def test_unknown_extension_returns_octet_stream(self):
        assert get_mime_type(Path("file.xyz123")) == "application/octet-stream"


# ── should_inline_preview ─────────────────────────────────────────────────────

class TestShouldInlinePreview:
    def test_pdf_is_inline(self):
        assert should_inline_preview("application/pdf") is True

    def test_plain_text_is_inline(self):
        assert should_inline_preview("text/plain") is True

    def test_html_is_inline(self):
        assert should_inline_preview("text/html") is True

    def test_csv_is_inline(self):
        assert should_inline_preview("text/csv") is True

    def test_json_is_inline(self):
        assert should_inline_preview("application/json") is True

    def test_png_is_inline(self):
        assert should_inline_preview("image/png") is True

    def test_jpeg_is_inline(self):
        assert should_inline_preview("image/jpeg") is True

    def test_svg_is_inline(self):
        assert should_inline_preview("image/svg+xml") is True

    def test_docx_is_not_inline(self):
        assert should_inline_preview(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ) is False

    def test_xlsx_is_not_inline(self):
        assert should_inline_preview(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ) is False

    def test_octet_stream_is_not_inline(self):
        assert should_inline_preview("application/octet-stream") is False

    def test_zip_is_not_inline(self):
        assert should_inline_preview("application/zip") is False


# ── find_file_in_storage ──────────────────────────────────────────────────────

class TestFindFileInStorage:
    """Uses a temporary directory patched as STORAGE_BASE so no real files needed."""

    @pytest.fixture()
    def fake_storage(self, tmp_path):
        """Patch STORAGE_BASE to a temp dir and create the standard sub-folders."""
        import backend.utils.file_server as fs_mod
        original = fs_mod.STORAGE_BASE
        fs_mod.STORAGE_BASE = tmp_path
        for folder in ["documents", "content", "images", "spreadsheets"]:
            (tmp_path / folder).mkdir(parents=True, exist_ok=True)
        yield tmp_path
        fs_mod.STORAGE_BASE = original

    def test_not_found_returns_none(self, fake_storage):
        result = find_file_in_storage("nonexistent.pdf")
        assert result is None

    def test_single_match_returns_path_and_relative(self, fake_storage):
        f = fake_storage / "documents" / "report.pdf"
        f.write_bytes(b"%PDF-1.4")
        result = find_file_in_storage("report.pdf")
        assert result is not None
        found_path, relative = result
        assert found_path == f
        assert "report.pdf" in relative

    def test_path_style_input_direct_lookup(self, fake_storage):
        """Filename containing '/' triggers direct resolve_and_validate_path lookup."""
        f = fake_storage / "images" / "photo.png"
        f.write_bytes(b"\x89PNG")
        result = find_file_in_storage("images/photo.png")
        assert result is not None
        found_path, relative = result
        assert found_path == f

    def test_path_style_nonexistent_returns_none(self, fake_storage):
        result = find_file_in_storage("documents/missing.pdf")
        assert result is None

    def test_multiple_matches_returns_newest(self, fake_storage):
        """When the same filename exists in multiple folders, newest mtime wins."""
        old_file = fake_storage / "documents" / "data.csv"
        new_file = fake_storage / "spreadsheets" / "data.csv"
        old_file.write_bytes(b"old")
        new_file.write_bytes(b"new")
        # Make the spreadsheets copy clearly newer
        old_time = old_file.stat().st_mtime - 100
        os.utime(old_file, (old_time, old_time))
        result = find_file_in_storage("data.csv")
        assert result is not None
        found_path, _ = result
        assert found_path == new_file

    def test_relative_path_in_result_is_string(self, fake_storage):
        f = fake_storage / "content" / "notes.txt"
        f.write_bytes(b"hello")
        result = find_file_in_storage("notes.txt")
        assert result is not None
        _, relative = result
        assert isinstance(relative, str)


# ── verify_clerk_token — generic-exception fallback ───────────────────────────

class TestVerifyClerkTokenGenericException:
    def test_unexpected_exception_from_decode_raises_401(self):
        """The catch-all except in verify_clerk_token converts any non-HTTP error to 401."""
        with patch(
            "backend.auth._decode_and_verify_token",
            side_effect=RuntimeError("unexpected internal error"),
        ):
            with pytest.raises(HTTPException) as exc:
                verify_clerk_token("Bearer sometoken")
        assert exc.value.status_code == 401
        assert "Invalid or expired token" in exc.value.detail
