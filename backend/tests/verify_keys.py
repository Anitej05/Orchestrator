
import os
import sys
from unittest.mock import patch

# Ensure backend root is in path
sys.path.insert(0, ".")

from backend.utils.key_manager import load_keys_from_env, KeyManager

def test_load_keys_single():
    print("Testing single key from env...")
    with patch.dict(os.environ, {"CEREBRAS_API_KEYS": "key1"}):
        keys = load_keys_from_env()
        assert keys == ["key1"]
    print("PASS")

def test_load_keys_multiple():
    print("Testing multiple keys from env...")
    with patch.dict(os.environ, {"CEREBRAS_API_KEYS": "key1,key2, key3 "}):
        keys = load_keys_from_env()
        assert keys == ["key1", "key2", "key3"]
    print("PASS")

def test_key_manager_init():
    print("Testing KeyManager init with env...")
    with patch.dict(os.environ, {"CEREBRAS_API_KEYS": "keyA,keyB"}):
        km = KeyManager()
        # Access internal _keys for verification
        assert km._keys == ["keyA", "keyB"]
        assert km.get_current_key() in ["keyA", "keyB"]
    print("PASS")

def test_key_rotation():
    print("Testing KeyManager rotation on rate limit...")
    with patch.dict(os.environ, {"CEREBRAS_API_KEYS": "k1,k2,k3"}):
        km = KeyManager()
        
        # 1. Initial key
        first_key = km.get_current_key()
        print(f"  Initial key: {first_key}")
        assert first_key in ["k1", "k2", "k3"]
        
        # 2. Rate limit the first key
        print(f"  Reporting rate limit for {first_key}")
        km.report_rate_limit(first_key)
        
        # 3. Should get a different key
        second_key = km.get_current_key()
        print(f"  Second key: {second_key}")
        assert second_key != first_key
        assert second_key in ["k1", "k2", "k3"]
        
        # 4. Rate limit the second key
        print(f"  Reporting rate limit for {second_key}")
        km.report_rate_limit(second_key)
        
        # 5. Should get the third key
        third_key = km.get_current_key()
        print(f"  Third key: {third_key}")
        assert third_key != first_key
        assert third_key != second_key
        assert third_key in ["k1", "k2", "k3"]
        
    print("PASS")

if __name__ == "__main__":
    try:
        test_load_keys_single()
        test_load_keys_multiple()
        test_key_manager_init()
        test_key_rotation()
        print("\nALL KEY TESTS PASSED ✅")
    except AssertionError as e:
        print(f"\nTEST FAILED ❌: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR ❌: {e}")
        sys.exit(1)
