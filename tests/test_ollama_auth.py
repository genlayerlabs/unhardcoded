"""Tests for Ollama Ed25519 OAuth authentication."""
import base64
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Skip all tests if cryptography not available
pytest.importorskip("cryptography")

from sources.ollama_auth import (
    get_ollama_auth_header,
    has_ollama_key,
    can_use_ed25519_auth,
    OllamaAuthError,
    _load_ed25519_key_ssh,
    _create_challenge,
    _sign_challenge,
    HAS_CRYPTOGRAPHY,
    OLLAMA_KEY_PATH,
)


class TestHasKeyFunctions:
    """Test key detection functions."""

    def test_has_ollama_key_exists(self, tmp_path):
        """Test has_ollama_key returns True when key exists."""
        key_path = tmp_path / "id_ed25519"
        key_path.write_bytes(b"dummy key content")
        assert has_ollama_key(key_path) is True

    def test_has_ollama_key_not_exists(self, tmp_path):
        """Test has_ollama_key returns False when key missing."""
        key_path = tmp_path / "missing_key"
        assert has_ollama_key(key_path) is False

    def test_can_use_ed25519_auth_no_key(self, tmp_path):
        """Test can_use_ed25519_auth returns False without key."""
        with patch("sources.ollama_auth.OLLAMA_KEY_PATH", tmp_path / "missing"):
            with patch("sources.ollama_auth.HAS_CRYPTOGRAPHY", True):
                from sources.ollama_auth import can_use_ed25519_auth
                # Reload to pick up patched path
                result = can_use_ed25519_auth()
                assert result is False


class TestChallengeCreation:
    """Test challenge string creation."""

    def test_challenge_get_no_body(self):
        """Test challenge for GET request with no body."""
        challenge = _create_challenge("GET", "https://ollama.com/api/tags", b"")
        # Challenge format: METHOD,URL,BASE64_HEX_SHA256(body)
        # SHA256 of empty string is hex string
        assert challenge.startswith("GET,")
        assert "https://ollama.com/api/tags," in challenge

    def test_challenge_with_body(self):
        """Test challenge with request body."""
        body = b'{"test": "data"}'
        challenge = _create_challenge("POST", "https://ollama.com/api/chat", body)
        assert challenge.startswith("POST,")
        assert "https://ollama.com/api/chat," in challenge


class TestAuthHeaderGeneration:
    """Test Authorization header generation."""

    @pytest.fixture
    def mock_key_path(self, tmp_path):
        """Create a mock Ed25519 key file."""
        # Generate a real Ed25519 key for testing
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization

        private_key = Ed25519PrivateKey.generate()

        # Serialize as SSH private key
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption()
        )

        key_path = tmp_path / "id_ed25519"
        key_path.write_bytes(private_bytes)
        return key_path

    def test_get_ollama_auth_header_success(self, mock_key_path):
        """Test successful auth header generation."""
        auth_header = get_ollama_auth_header(
            method="GET",
            url="https://ollama.com/api/tags",
            key_path=mock_key_path
        )

        # Header should be pubkey_b64:signature_b64
        assert ":" in auth_header
        parts = auth_header.split(":")
        assert len(parts) == 2

        # Both parts should be valid base64
        try:
            base64.b64decode(parts[0])
            base64.b64decode(parts[1])
        except Exception:
            pytest.fail("Auth header parts are not valid base64")

    def test_get_ollama_auth_header_missing_key(self, tmp_path):
        """Test auth header fails with missing key."""
        missing_key = tmp_path / "missing_key"

        with pytest.raises(OllamaAuthError) as exc_info:
            get_ollama_auth_header(key_path=missing_key)

        assert "not found" in str(exc_info.value)

    def test_auth_header_varies_by_timestamp(self, mock_key_path):
        """Test that auth headers vary over time (due to timestamp)."""
        import time

        # Generate first header
        header1 = get_ollama_auth_header(
            method="GET",
            url="https://ollama.com/api/tags",
            key_path=mock_key_path
        )

        # Wait for timestamp to change
        time.sleep(1.1)

        # Generate second header
        header2 = get_ollama_auth_header(
            method="GET",
            url="https://ollama.com/api/tags",
            key_path=mock_key_path
        )

        # Both should have valid format
        assert ":" in header1
        assert ":" in header2

        # Signatures should differ due to different timestamps
        # Format is pubkey:signature, so compare signatures (second part)
        sig1 = header1.split(":")[1]
        sig2 = header2.split(":")[1]
        assert sig1 != sig2, "Signatures should differ for different timestamps"


class TestKeyLoading:
    """Test Ed25519 key loading."""

    @pytest.fixture
    def real_key(self, tmp_path):
        """Create a real Ed25519 key for testing."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization

        private_key = Ed25519PrivateKey.generate()
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption()
        )

        key_path = tmp_path / "id_ed25519"
        key_path.write_bytes(private_bytes)
        return key_path

    def test_load_ed25519_key_ssh(self, real_key):
        """Test loading Ed25519 key from SSH format."""
        private_bytes, public_bytes = _load_ed25519_key_ssh(real_key)

        # Ed25519 keys are 32 bytes each
        assert len(private_bytes) == 32
        assert len(public_bytes) == 32

    def test_load_invalid_key(self, tmp_path):
        """Test loading invalid key raises error."""
        key_path = tmp_path / "invalid_key"
        key_path.write_bytes(b"not a valid key")

        with pytest.raises(OllamaAuthError):
            _load_ed25519_key_ssh(key_path)


class TestSignChallenge:
    """Test challenge signing."""

    @pytest.fixture
    def key_pair(self):
        """Generate a key pair for testing."""
        if not HAS_CRYPTOGRAPHY:
            pytest.skip("cryptography library not available")

        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization

        private_key = Ed25519PrivateKey.generate()
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        return private_bytes, public_bytes

    def test_sign_challenge_format(self, key_pair):
        """Test that signed challenge has correct format."""
        private_bytes, _ = key_pair
        challenge = "GET,https://ollama.com/api/tags,abc123"

        signature = _sign_challenge(private_bytes, challenge)

        # Should be pubkey_b64:signature_b64
        assert ":" in signature
        parts = signature.split(":")
        assert len(parts) == 2

        # Verify both parts are valid base64
        pubkey = base64.b64decode(parts[0])
        sig = base64.b64decode(parts[1])

        # Ed25519 public key is 32 bytes, signature is 64 bytes
        assert len(pubkey) == 32
        assert len(sig) == 64


class TestCloudAuthPreference:
    """Test auth method preference (Ed25519 vs API key)."""

    def test_ed25519_preferred_over_api_key(self, tmp_path):
        """Test that Ed25519 is preferred when both available."""
        # This tests the auth resolution logic in ollama.py
        from sources.ollama import OllamaSource

        # Create a mock key
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization

        private_key = Ed25519PrivateKey.generate()
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption()
        )
        key_path = tmp_path / "id_ed25519"
        key_path.write_bytes(private_bytes)

        # Create source with both Ed25519 key and API key
        with patch("sources.ollama_auth.OLLAMA_KEY_PATH", key_path):
            source = OllamaSource(
                catalog={},
                env_get=lambda k: "test-api-key" if k == "OLLAMA_API_KEY" else None
            )

            # Should have Ed25519 auth available
            assert source._use_ed25519 is True
            assert source._api_key == "test-api-key"

    def test_api_key_used_when_no_ed25519(self):
        """Test that API key is used when Ed25519 not available."""
        from sources.ollama import OllamaSource

        # Mock can_use_ed25519_auth to return False
        with patch("sources.ollama.can_use_ed25519_auth", return_value=False):
            source = OllamaSource(
                catalog={},
                env_get=lambda k: "test-api-key" if k == "OLLAMA_API_KEY" else None
            )

            # Without Ed25519 key, should use API key
            assert source._use_ed25519 is False
            assert source._api_key == "test-api-key"

    def test_no_auth_available(self):
        """Test when neither Ed25519 nor API key available."""
        from sources.ollama import OllamaSource

        with patch("sources.ollama_auth.has_ollama_key", return_value=False):
            source = OllamaSource(
                catalog={},
                env_get=lambda k: None
            )

            # Should have neither
            assert source._use_ed25519 is False
            assert source._api_key is None