import datetime, json, urllib.request, urllib.parse, pathlib

cred_file = pathlib.Path.home() / '.glacierguard' / 'cdse_credentials.json'
creds = json.loads(cred_file.read_text())
cid, csec = creds['client_id'], creds['client_secret']

token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
payload = urllib.parse.urlencode({'grant_type': 'client_credentials', 'client_id': cid, 'client_secret': csec}).encode()
req = urllib.request.Request(token_url, data=payload, headers={'Content-Type': 'application/x-www-form-urlencoded'})
with urllib.request.urlopen(req) as resp:
    token = json.loads(resp.read().decode())['access_token']

print("Token acquired successfully.")

now_utc = datetime.datetime.now(datetime.timezone.utc)
dt_90 = (now_utc - datetime.timedelta(days=90)).strftime('%Y-%m-%dT00:00:00Z')
dt_180 = (now_utc - datetime.timedelta(days=180)).strftime('%Y-%m-%dT00:00:00Z')
dt_now = now_utc.strftime('%Y-%m-%dT23:59:59Z')

lakes = [
    {'lake_id': 'imja_tsho', 'display_name': 'Imja Tsho', 'bbox': [86.900, 27.885, 86.955, 27.930]},
    {'lake_id': 'thulagi', 'display_name': 'Thulagi Glacier Lake', 'bbox': [84.460, 28.465, 84.515, 28.515]},
    {'lake_id': 'tsho_rolpa', 'display_name': 'Tsho Rolpa', 'bbox': [86.455, 27.840, 86.515, 27.890]},
    {'lake_id': 'lower_barun', 'display_name': 'Lower Barun', 'bbox': [87.060, 27.825, 87.110, 27.865]},
    {'lake_id': 'sabai_tsho', 'display_name': 'Sabai Tsho', 'bbox': [86.600, 27.775, 86.645, 27.815]},
]

stac_url = 'https://catalogue.dataspace.copernicus.eu/stac/search'
auth_hdr = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

lake_scenes = {}
for lk in lakes:
    lid = lk['lake_id']
    bbox = lk['bbox']
    lake_scenes[lid] = {'s1': [], 's2': []}
    print(f"\n--- Searching {lk['display_name']} ({lid}) ---")
    
    # S1
    p_s1 = json.dumps({'collections': ['sentinel-1-grd'], 'bbox': bbox, 'datetime': f"{dt_90}/{dt_now}", 'limit': 30}).encode()
    r_s1 = urllib.request.Request(stac_url, data=p_s1, headers=auth_hdr)
    with urllib.request.urlopen(r_s1) as resp:
        s1_res = json.loads(resp.read().decode())
    s1_feats = s1_res.get('features', [])
    print(f"  S1 (90d): found {len(s1_feats)} scenes")
    # Sort descending by date
    s1_feats.sort(key=lambda f: f.get('properties', {}).get('datetime', ''), reverse=True)
    lake_scenes[lid]['s1'] = s1_feats
    for f in s1_feats[:3]:
        props = f.get('properties', {})
        print(f"    S1: {f['id'][:40]} | DT: {props.get('datetime', '')[:19]} | Orbit: {props.get('sat:relative_orbit', props.get('relativeOrbitNumber', 'N/A'))} {props.get('sat:orbit_state', '')}")
    
    # S2
    p_s2 = json.dumps({'collections': ['sentinel-2-l2a'], 'bbox': bbox, 'datetime': f"{dt_180}/{dt_now}", 'limit': 50}).encode()
    r_s2 = urllib.request.Request(stac_url, data=p_s2, headers=auth_hdr)
    with urllib.request.urlopen(r_s2) as resp:
        s2_res = json.loads(resp.read().decode())
    s2_feats = s2_res.get('features', [])
    # Sort descending by date
    s2_feats.sort(key=lambda f: f.get('properties', {}).get('datetime', ''), reverse=True)
    print(f"  S2 (180d): found {len(s2_feats)} scenes")
    lake_scenes[lid]['s2'] = s2_feats
    for f in s2_feats[:3]:
        props = f.get('properties', {})
        print(f"    S2: {f['id'][:40]} | DT: {props.get('datetime', '')[:10]} | Cloud: {props.get('eo:cloud_cover', 'N/A')}% | Tile: {props.get('grid:code', props.get('s2:mgrs_tile', ''))}")

pathlib.Path('scratch/stac_5_lakes.json').write_text(json.dumps({lid: {'s1': [f['id'] for f in lake_scenes[lid]['s1']], 's2': [f['id'] for f in lake_scenes[lid]['s2']]} for lid in lake_scenes}, indent=2))
print("\nSaved scene IDs to scratch/stac_5_lakes.json")
