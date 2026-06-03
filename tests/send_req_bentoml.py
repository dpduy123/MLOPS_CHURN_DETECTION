import requests
import json

BENTO_URL = "http://localhost:3000"

# ─── Test cases ────────────────────────────────────────────────────────────────
COLUMNS = [
    "CustomerID", "Age", "Gender", "Tenure", "Usage Frequency",
    "Support Calls", "Payment Delay", "Subscription Type",
    "Contract Length", "Total Spend", "Last Interaction"
]

SAMPLES = [
    [2.0, 30.0, "Female", 39.0, 14.0,  5.0, 18.0, "Standard",   "Annual",    932.0, 17.0],
    [3.0, 65.0, "Female", 49.0,  1.0, 10.0,  8.0, "Basic",      "Monthly",   557.0,  6.0],
    [4.0, 55.0, "Female", 14.0,  4.0,  6.0, 18.0, "Basic",      "Quarterly", 185.0,  3.0],
]

# ─── Helpers ───────────────────────────────────────────────────────────────────
def print_sep(title=""):
    print(f"\n{'─' * 50}")
    if title:
        print(f"  {title}")
        print(f"{'─' * 50}")

def test_healthz():
    print_sep("Health check")
    r = requests.post(f"{BENTO_URL}/healthz")
    print(f"Status : {r.status_code}")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    assert r.status_code == 200, f"Healthz failed: {r.text}"
    print("✅ OK")

def test_predict_batch():
    print_sep("Predict — 3 rows")
    payload = {
        "input": {
            "dataframe_split": {
                "columns": COLUMNS,
                "data": SAMPLES
            }
        }
    }
    r = requests.post(
        f"{BENTO_URL}/predict",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    print(f"Status : {r.status_code}")
    assert r.status_code == 200, f"Predict failed: {r.text}"

    result = r.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))

    for i, (pred, label) in enumerate(zip(result["predictions"], result["labels"])):
        print(f"  Row {i+1}: {pred} → {label}")

    assert result["row_count"] == len(SAMPLES)
    print("✅ OK")

def test_predict_single():
    print_sep("Predict — 1 row")
    payload = {
        "input": {
            "dataframe_split": {
                "columns": COLUMNS,
                "data": [SAMPLES[0]]
            }
        }
    }
    r = requests.post(
        f"{BENTO_URL}/predict",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    assert r.status_code == 200, f"Predict failed: {r.text}"
    result = r.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  Prediction: {result['predictions'][0]} → {result['labels'][0]}")
    assert result["row_count"] == 1
    print("✅ OK")

# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Target: {BENTO_URL}")
    try:
        test_healthz()
        test_predict_single()
        test_predict_batch()
        print_sep()
        print("  ✅ Tất cả test passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Không kết nối được tới {BENTO_URL} — service có đang chạy không?")