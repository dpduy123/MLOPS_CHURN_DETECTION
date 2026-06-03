import time, json, urllib.request
ns = int(time.time() * 1e9)
body = json.dumps({"streams":[{"stream":{"service":"churn-prediction"},"values":[[str(ns),"test log from quickpush"]]}]})
req = urllib.request.Request("http://localhost:3100/loki/api/v1/push", data=body.encode(), headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=5) as r:
    print(r.read().decode())
