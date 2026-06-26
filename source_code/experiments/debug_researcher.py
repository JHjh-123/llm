import urllib.request
import json
import urllib.error

mock_message = {
    "message_id": "msg_test_001",
    "task_id": "task_test_001",
    "parent_id": None,
    "sender": "test_runner",
    "receiver": "agent",
    "mode": "structured",
    "content": "This is a test content representing intermediate agent outcomes."
}

req = urllib.request.Request(
    'http://localhost:8000/agent/researcher',
    data=json.dumps({
        'task': 'Design a memory system schema.',
        'plan': mock_message,
        'task_id': 'task_test_001',
        'mode': 'structured'
    }).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
try:
    resp = urllib.request.urlopen(req)
    print("Success:", resp.read().decode())
except urllib.error.HTTPError as e:
    print("Status code:", e.code)
    print("Response body:", e.read().decode())
except Exception as e:
    print("Other error:", e)
