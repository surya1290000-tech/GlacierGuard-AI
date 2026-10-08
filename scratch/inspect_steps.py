import json

with open(r'C:\Users\ASUS\.gemini\antigravity-ide\brain\56853e18-ea51-4fdd-8b62-0dcad1d05fd5\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 420 <= i <= 445:
            d = json.loads(line)
            print(f"[{i}] step={d.get('step_index')}, type={d.get('type')}, source={d.get('source')}")
            if 'content' in d and d['content']:
                print('  content:', repr(d['content'][:150]))
            if 'tool_calls' in d:
                for tc in d['tool_calls']:
                    print('  tc:', tc.get('name'), tc.get('args', {}).get('toolSummary'))
