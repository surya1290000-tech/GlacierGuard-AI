import datetime, json, urllib.request, urllib.parse, pathlib

cred_file = pathlib.Path.home() / '.glacierguard' / 'cdse_credentials.json'
creds = json.loads(cred_file.read_text())
cid, csec = creds['client_id'], creds['client_secret']

token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
payload = urllib.parse.urlencode({
    'grant_type': 'client_credentials',
    'client_id': cid, 'client_secret': csec
}).encode()
req = urllib.request.Request(token_url, data=payload, headers={'Content-Type': 'application/x-www-form-urlencoded'})
with urllib.request.urlopen(req) as resp:
    token = json.loads(resp.read().decode())['access_token']

now_utc = datetime.datetime.now(datetime.timezone.utc)
dt_90 = (now_utc - datetime.timedelta(days=90)).strftime('%Y-%m-%dT00:00:00Z')
dt_now = now_utc.strftime('%Y-%m-%dT23:59:59Z')
dt_range_90 = f'{dt_90}/{dt_now}'

bbox = [86.900, 27.885, 86.955, 27.930]
stac_url = 'https://catalogue.dataspace.copernicus.eu/stac/search'
auth_hdr = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

print("=== SENTINEL-1 SCENES ===")
p_s1 = json.dumps({'collections': ['sentinel-1-grd'], 'bbox': bbox, 'datetime': dt_range_90, 'limit': 50}).encode()
r_s1 = urllib.request.Request(stac_url, data=p_s1, headers=auth_hdr)
with urllib.request.urlopen(r_s1) as resp:
    s1_res = json.loads(resp.read().decode())

s1_features = s1_res.get('features', [])
print(f"Total S1: {len(s1_features)}")
for f in s1_features[:15]:
    props = f.get('properties', {})
    pid = f.get('id', '')
    dt = props.get('datetime', '')
    orbit_dir = props.get('sat:orbit_state', props.get('orbitDirection', 'N/A'))
    rel_orbit = props.get('sat:relative_orbit', props.get('relativeOrbitNumber', 'N/A'))
    pols = props.get('polarization', props.get('sar:polarizations', 'N/A'))
    mode = props.get('sar:instrument_mode', 'N/A')
    print(f"S1 ID: {pid} | DT: {dt} | Orbit: {orbit_dir} (rel {rel_orbit}) | Pols: {pols} | Mode: {mode}")

print("\n=== SENTINEL-2 SCENES ===")
p_s2 = json.dumps({'collections': ['sentinel-2-l2a'], 'bbox': bbox, 'datetime': dt_range_90, 'limit': 50}).encode()
r_s2 = urllib.request.Request(stac_url, data=p_s2, headers=auth_hdr)
with urllib.request.urlopen(r_s2) as resp:
    s2_res = json.loads(resp.read().decode())

s2_features = s2_res.get('features', [])
print(f"Total S2: {len(s2_features)}")
for f in s2_features[:20]:
    props = f.get('properties', {})
    pid = f.get('id', '')
    dt = props.get('datetime', '')
    cloud = props.get('eo:cloud_cover', 'N/A')
    tile = props.get('grid:code', props.get('s2:mgrs_tile', 'N/A'))
    print(f"S2 ID: {pid} | DT: {dt} | Cloud: {cloud}% | Tile: {tile}")
