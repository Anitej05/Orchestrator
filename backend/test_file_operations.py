import requests
import json
import time

def test_download():
    """Test file download functionality"""
    url = "http://localhost:8070/browse"
    payload = {
        "task": "Go to httpbin.org/image/png and download the image",
        "thread_id": "test-download",
        "max_steps": 10
    }
    
    print("\n" + "="*80)
    print("TEST: File Download")
    print("="*80)
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result['success']}")
            print(f"📝 Summary: {result['task_summary'][:120]}")
            print(f"📥 Downloads: {len(result.get('downloaded_files', []))}")
            
            if result.get('downloaded_files'):
                for df in result['downloaded_files']:
                    print(f"   - {df}")
            
            return result['success']
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality to ensure nothing broke"""
    url = "http://localhost:8070/browse"
    payload = {
        "task": "Go to example.com and tell me what it says",
        "thread_id": "test-basic",
        "max_steps": 5
    }
    
    print("\n" + "="*80)
    print("TEST: Basic Functionality (Regression Test)")
    print("="*80)
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result['success']}")
            print(f"📝 Summary: {result['task_summary'][:120]}")
            print(f"📊 Metrics: {result.get('metrics', {})}")
            return result['success']
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("FILE OPERATIONS TEST SUITE")
    print("="*80)
    
    tests = []
    
    # Test basic functionality first
    tests.append(("Basic Functionality", test_basic_functionality()))
    time.sleep(2)
    
    # Test download
    tests.append(("File Download", test_download()))
    
    print("\n" + "="*80)
    passed = sum(1 for _, result in tests if result)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("="*80 + "\n")
    
    if passed == len(tests):
        print("🎉 All file operation tests passed!")
    else:
        print("⚠️  Some tests failed")
