import os
import re

patterns = [
    re.compile(r'(cdse_client_secret|client_secret|cdsapi_key|api_key|password)\s*[:=]\s*["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'bearer\s+([a-zA-Z0-9\-_]{25,})', re.IGNORECASE),
    re.compile(r'(sh-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', re.IGNORECASE),
]

# Patterns that are placeholders or safe
safe_values = {'', 'your_key', 'your_secret', 'your_client_id', 'your_client_secret', 'none', 'placeholder', '<placeholder>', 'test'}

extensions_to_check = {'.py', '.ipynb', '.md', '.json', '.yml', '.yaml', '.toml', '.txt', '.sh', '.env'}

findings = []
for root, dirs, files in os.walk('.'):
    # Skip directories that should never be scanned/committed
    if any(skip in root for skip in ['.git', '.venv', 'raw', 'cache', '__pycache__', '.ipynb_checkpoints']):
        continue
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in extensions_to_check:
            filepath = os.path.join(root, f)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as fp:
                    content = fp.read()
                    for pat in patterns:
                        for match in pat.finditer(content):
                            val = match.group(0)
                            # Check if it's an environment variable lookup or comment
                            if 'os.environ' in val or 'os.getenv' in val or 'CDSE_CLIENT_SECRET' in val and 'None' in val:
                                continue
                            findings.append((filepath, match.start(), val[:50]))
            except Exception as e:
                pass

print(f"Total potential secret findings: {len(findings)}")
for path, pos, sample in findings:
    print(f"  {path} (pos {pos}): {sample}")
