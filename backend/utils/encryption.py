# backend/utils/encryption.py
"""
Encryption utilities for secure credential storage.
Uses Fernet symmetric encryption from the cryptography library.
"""

import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

# Get encryption key from environment or generate one
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    # Generate a key if not set (for development only)
    # In production, this should be set in environment variables
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    print(f"WARNING: No ENCRYPTION_KEY found. Generated temporary key: {ENCRYPTION_KEY}")
    print("Add this to your .env file: ENCRYPTION_KEY=" + ENCRYPTION_KEY)

cipher_suite = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)


def encrypt(plaintext: str) -> str:
    """
    Encrypt a plaintext string.
    
    Args:
        plaintext: The string to encrypt
        
    Returns:
        Base64-encoded encrypted string
    """
    if not plaintext:
        return ""
    
    encrypted_bytes = cipher_suite.encrypt(plaintext.encode())
    return encrypted_bytes.decode()


def decrypt(ciphertext: str) -> str:
    """
    Decrypt an encrypted string.
    
    Args:
        ciphertext: The encrypted string to decrypt
        
    Returns:
        Decrypted plaintext string
    """
    if not ciphertext:
        return ""
    
    decrypted_bytes = cipher_suite.decrypt(ciphertext.encode())
    return decrypted_bytes.decode()
