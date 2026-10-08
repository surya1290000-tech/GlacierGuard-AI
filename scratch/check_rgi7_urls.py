import requests, re

# Check if the global attributes CSV has what we need
# Also check other potential sources
urls = [
    "https://cluster.klima.uni-bremen.de/~oggm/rgi/www.glims.org/",
    "https://cluster.klima.uni-bremen.de/~oggm/rgi/www.glims.org/RGI/",
]

for url in urls:
    print(f"Checking: {url}")
    try:
        r = requests.get(url, timeout=15)
        print(f"  Status: {r.status_code}")
        if r.ok:
            links = re.findall(r'href="([^"]*)"', r.text)
            for l in links:
                lo = l.lower()
                if "15" in lo or "rgi7" in lo or "rgi2000" in lo or "south" in lo or "zip" in lo:
                    print(f"    MATCH: {l}")
            if not any("15" in l or "rgi7" in l.lower() for l in links):
                for l in links[:25]:
                    print(f"    {l}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

# Also try NSIDC-like pattern via GLIMS 
print("Trying GLIMS/NSIDC patterns...")
for url in [
    "https://www.glims.org/RGI/rgi70_dl.html",
    "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-G/",
]:
    print(f"  {url}")
    try:
        r = requests.get(url, timeout=15, allow_redirects=True)
        print(f"    Status: {r.status_code}")
        if r.ok and len(r.text) < 50000:
            links = re.findall(r'href="([^"]*)"', r.text)
            for l in links:
                if "15" in l or "south" in l.lower():
                    print(f"    MATCH: {l}")
    except Exception as e:
        print(f"    Error: {e}")
