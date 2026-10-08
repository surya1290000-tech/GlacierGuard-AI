"""
NB10 v2 builder - secure credential layer
"""
import json, uuid, pathlib, textwrap

ROOT    = pathlib.Path(".").resolve()
NB_PATH = ROOT / "notebooks" / "10_nrt_lake_monitoring.ipynb"

def md(text):
    src = textwrap.dedent(text).strip()
    return {"cell_type":"markdown","metadata":{},"source":(src+"\n").splitlines(keepends=True),"id":str(uuid.uuid4())[:8]}

def code(text):
    src = textwrap.dedent(text).strip()
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":(src+"\n").splitlines(keepends=True),"id":str(uuid.uuid4())[:8]}

cells = []

cells.append(md("""
# GlacierGuard-AI -- Notebook 10
## Near-Real-Time Sentinel-1 / Sentinel-2 Lake Monitoring

**Project**: GlacierGuard-AI -- Autonomous GLOF Early Warning System
**Infrastructure**: Copernicus Data Space Ecosystem (CDSE) / Sentinel Hub Processing API

> **Security**
> - Credentials are NEVER printed, logged, committed, or embedded in cells.
> - Credential resolution order (2-tier):
>   1. Environment variables CDSE_CLIENT_ID / CDSE_CLIENT_SECRET
>   2. User-local file ~/.glacierguard/cdse_credentials.json (outside Git repo)
>
> To populate the credential file from PowerShell (run once):
>   .venv\\Scripts\\python.exe scratch\\setup_cdse_credentials.py
"""))

print(f"Builder running OK, cells so far: {len(cells)}")
