"""
Ollama Ed25519 OAuth authentication.

Implements the challenge-response authentication flow used by Ollama Cloud.
Based on the Ollama CLI authentication mechanism using Ed25519 SSH keys.

Key file: ~/.ollama/id_ed25519 (SSH format Ed25519 private key)
Challenge format: METHOD,URL,BASE64_HEX_SHA256(body)
Authorization header: <pubkey_b64>:<signature_b64>
"""
from __future__ import annotations

import base64
import hashlib
import os
import time
from pathlib import Path
from typing import Optional

# Ed25519 support - cryptography library preferred, fallback to nacl
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    try:
        from nacl.signing import SigningKey
        from nacl.encoding import Base64Encoder
        HAS_NACL = True
    except ImportError:
        HAS_NACL = False


OLLAMA_KEY_PATH = Path.home() / ".ollama" / "id_ed25519"


class OllamaAuthError(Exception):
    """Raised when Ollama authentication fails."""
    pass


def _load_ed25519_key_ssh(key_path: Path) -> tuple[bytes, bytes]:
    """
    Load Ed25519 key from SSH private key file.

    Returns (private_key_bytes, public_key_bytes).
    Raises OllamaAuthError if key cannot be loaded.
    """
    if not key_path.exists():
        raise OllamaAuthError(f"Ollama key not found at {key_path}")

    try:
        key_data = key_path.read_bytes()
    except (PermissionError, OSError) as e:
        raise OllamaAuthError(f"Cannot read Ollama key at {key_path}: {e}")

    if HAS_CRYPTOGRAPHY:
        return _load_with_cryptography(key_data)
    elif HAS_NACL:
        return _load_with_nacl(key_data)
    else:
        raise OllamaAuthError(
            "No Ed25519 library available. Install 'cryptography' or 'pynacl'."
        )


def _load_with_cryptography(key_data: bytes) -> tuple[bytes, bytes]:
    """Load Ed25519 key using cryptography library."""
    try:
        private_key = serialization.load_ssh_private_key(
            key_data,
            password=None,
            backend=default_backend()
        )
    except (ValueError, Exception) as e:
        raise OllamaAuthError(f"Failed to parse SSH key: {e}")

    if not isinstance(private_key, Ed25519PrivateKey):
        raise OllamaAuthError("Key is not an Ed25519 key")

    # Get private key bytes (32 bytes seed)
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )

    # Get public key bytes (32 bytes)
    public_key = private_key.public_key()
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )

    return private_bytes, public_bytes


def _load_with_nacl(key_data: bytes) -> tuple[bytes, bytes]:
    """Load Ed25519 key using PyNaCl library.

    Note: This is a fallback parser. The cryptography library is preferred
    for full OpenSSH format support.
    """
    # Parse SSH private key format
    lines = key_data.decode('utf-8').strip().split('\n')

    if not lines[0].startswith('-----BEGIN') or 'PRIVATE KEY' not in lines[0]:
        raise OllamaAuthError("Invalid SSH key format")

    # Decode base64 content
    b64_content = ''.join(lines[1:-1])
    key_bytes = base64.b64decode(b64_content)

    try:
        # Check for OpenSSH v1 format (modern Ed25519 keys)
        # Magic: "openssh-key-v1\x00"
        if key_bytes.startswith(b"openssh-key-v1\x00"):
            # This is the modern OpenSSH format - complex parsing required
            # Recommend using cryptography library for this
            raise OllamaAuthError(
                "OpenSSH v1 key format requires 'cryptography' library. "
                "Install it with: pip install cryptography"
            )

        # Legacy PEM format - try to extract raw 64 bytes (seed + public)
        if len(key_bytes) >= 64:
            # For legacy Ed25519 PEM format, last 64 bytes are seed + public
            seed = key_bytes[-64:][:32]
            public = key_bytes[-32:]  # Last 32 bytes are public key
            return seed, public

        raise OllamaAuthError("Cannot parse SSH key with PyNaCl - install 'cryptography'")
    except OllamaAuthError:
        raise
    except Exception as e:
        raise OllamaAuthError(f"Failed to parse SSH key: {e}")


def _create_challenge(method: str, url: str, body: bytes = b"") -> str:
    """
    Create Ollama challenge string.

    Format: METHOD,URL,BASE64_HEX_SHA256(body)

    For GET requests with no body, the hash is of empty string.
    """
    # SHA256 of body, hex encoded, then base64
    sha256_hash = hashlib.sha256(body).hexdigest()
    body_hash_b64 = base64.b64encode(sha256_hash.encode()).decode()

    return f"{method},{url},{body_hash_b64}"


def _sign_challenge(private_bytes: bytes, challenge: str) -> str:
    """
    Sign the challenge using Ed25519 and return the auth token.

    Returns: <pubkey_b64>:<signature_b64>
    """
    if HAS_CRYPTOGRAPHY:
        return _sign_with_cryptography(private_bytes, challenge)
    elif HAS_NACL:
        return _sign_with_nacl(private_bytes, challenge)
    else:
        raise OllamaAuthError("No Ed25519 signing library available")


def _sign_with_cryptography(private_bytes: bytes, challenge: str) -> str:
    """Sign using cryptography library."""
    private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
    public_key = private_key.public_key()

    # Sign the challenge
    signature = private_key.sign(challenge.encode())

    # Get public key bytes
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )

    # Format: pubkey_b64:signature_b64
    pubkey_b64 = base64.b64encode(public_bytes).decode()
    sig_b64 = base64.b64encode(signature).decode()

    return f"{pubkey_b64}:{sig_b64}"


def _sign_with_nacl(private_bytes: bytes, challenge: str) -> str:
    """Sign using PyNaCl library."""
    signing_key = SigningKey(private_bytes)
    signed = signing_key.sign(challenge.encode())

    # PyNaCl signed message includes signature + message
    # We need just the signature (64 bytes) and public key
    signature = signed.signature
    public_bytes = bytes(signing_key.verify_key)

    pubkey_b64 = base64.b64encode(public_bytes).decode()
    sig_b64 = base64.b64encode(signature).decode()

    return f"{pubkey_b64}:{sig_b64}"


def get_ollama_auth_header(
    method: str = "GET",
    url: str = "",
    body: bytes = b"",
    key_path: Optional[Path] = None
) -> str:
    """
    Generate Authorization header for Ollama Cloud using Ed25519 OAuth.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Target URL (e.g., "https://ollama.com/api/tags")
        body: Request body (for POST requests)
        key_path: Path to Ed25519 key (default: ~/.ollama/id_ed25519)

    Returns:
        Authorization header value (e.g., "pubkey_b64:signature_b64")

    Raises:
        OllamaAuthError: If key not found or signing fails
    """
    key_path = key_path or OLLAMA_KEY_PATH

    # Load the Ed25519 key
    private_bytes, _ = _load_ed25519_key_ssh(key_path)

    # Create challenge with timestamp
    # Ollama uses: METHOD,URL,ts=timestamp
    ts = int(time.time())
    challenge_with_ts = f"{method},{url}?ts={ts}"

    # Add body hash if present
    if body:
        sha256_hash = hashlib.sha256(body).hexdigest()
        body_hash_b64 = base64.b64encode(sha256_hash.encode()).decode()
        challenge_with_ts = f"{method},{url}?ts={ts},{body_hash_b64}"

    # Sign and return
    return _sign_challenge(private_bytes, challenge_with_ts)


def has_ollama_key(key_path: Optional[Path] = None) -> bool:
    """Check if Ollama Ed25519 key exists."""
    return (key_path or OLLAMA_KEY_PATH).exists()


def can_use_ed25519_auth() -> bool:
    """Check if Ed25519 authentication is available (key + library)."""
    return has_ollama_key() and (HAS_CRYPTOGRAPHY or HAS_NACL)