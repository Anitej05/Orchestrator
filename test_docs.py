import requests

BASE_URL = "http://localhost:8050"

def test_analyze():
    payload = {
        "file_path": r"d:\Internship\Orbimesh\storage\documents\test_script_doc.docx",
        "query": "What is this document about?"
    }
    res = requests.post(f"{BASE_URL}/analyze", json=payload)
    print(f"POST /analyze -> {res.status_code}")
    print(res.json())

def test_extract():
    payload = {
        "file_path": r"d:\Internship\Orbimesh\storage\documents\test_script_doc.docx",
        "extraction_type": "text"
    }
    res = requests.post(f"{BASE_URL}/extract", json=payload)
    print(f"POST /extract -> {res.status_code}")
    print(res.json())

if __name__ == "__main__":
    test_analyze()
    test_extract()
