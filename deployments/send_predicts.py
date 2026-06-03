import urllib.request, json, time
url = 'http://localhost:3000/predict'
headers = {'Content-Type':'application/json'}
COLUMNS = ["CustomerID", "Age", "Gender", "Tenure", "Usage Frequency","Support Calls", "Payment Delay", "Subscription Type","Contract Length", "Total Spend", "Last Interaction"]
SAMPLE = [2.0, 30.0, "Female", 39.0, 14.0,  5.0, 18.0, "Standard",   "Annual",    932.0, 17.0]
payload = {"input":{"dataframe_split":{"columns":COLUMNS, "data":[SAMPLE]}}}
for i in range(10):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            print(i, r.status)
    except Exception as e:
        print('err', e)
    time.sleep(1)
