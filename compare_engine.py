# compare_engine.py
"""
Updated compare engine (backend) for OLD vs NEW CSV comparison.

Features:
- Retains your original matching logic (address-first fuzzy matching, phone/geo corroboration,
  store-id fallback, schema projection).
- Provides in-memory entrypoints intended for Streamlit or other UIs:
    - compare_pair_df(old_df, new_df) -> DataFrame (no file writes)
    - compare_uploaded_files(old_file, new_file) -> DataFrame (accepts file-like or path)
    - compare_files_paths(old_path, new_path) -> DataFrame (reads from paths; returns DF)
- Keeps helper functions for CLI/batch usage (run_batch, safe_write_csv) if you want to use them,
  but the UI wrappers never write to disk.
- No code runs on import.
"""
import math
import re
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Set, Tuple
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

# ===============================
# Folders (relative to this file) - used only by batch helpers (optional)
# ===============================
BASE_DIR = Path(__file__).resolve().parent
OLD_DIR = BASE_DIR / "old"
NEW_DIR = BASE_DIR / "new"
OUT_DIR = BASE_DIR / "compared_data"
# NOTE: we will create OUT_DIR lazily in safe_write_csv if needed, so importing this module
# doesn't force directory creation in environments where you don't want writes.

# ===============================
# Tunables (adjustable at runtime)
# ===============================
PAIRING_STRATEGY = "closest-prev"  # or "same-date"
MAX_METERS = 200                   # geo tolerance for corroboration/tie-break
ADDR_STRICT = 0.90                 # strict fuzzy threshold for address acceptance
ADDR_LOOSE  = 0.75                 # looser threshold if phone OR geo corroborate
ENCODING = "utf-8-sig"

# Never overwrite these from NEW on matched rows:
DO_NOT_OVERWRITE = {"city", "state", "zip", "zipcode", "zip_code", "postal", "postalcode", "postal_code"}

# Column candidate names (case-insensitive)
ADDR_CANDS  = [
    "address","address1","address_1","address line 1","address line1","addr","addr1",
    "street","street1","address_line","address_line1","address_line_1","street_address",
    "streetaddress","line1","address2","address_2","address line 2","address_line_2"
]
CITY_CANDS  = ["city","town"]
STATE_CANDS = ["state","province","region","state_code","st"]
ZIP_CANDS   = ["zip","zipcode","zip_code","postal","postalcode","postal_code"]
NAME_CANDS  = ["store_name","name","dba","distributor_name","distributor"]    # for reporting/mapping
PHONE_CANDS = ["phone","phone_number","phonenumber","telephone","tel"]
ID_CANDS = [
    "store_number","store number","storenumber","store_num","store num","storenum",
    "storeid","store id","store_id","store code","store_code","storecode",
    "location_id","location id","locationid","location_number","location number","locationnumber",
    "site_id","site id","siteid","id","code"
]

# Date tokens in filenames
DATE_TOKEN_FINDER = re.compile(r"(\d{2}-\d{2}-\d{2}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})")
# Brand = prefix before first underscore (e.g., "taco_bell_*" -> "taco")
BRAND_RE  = re.compile(r"^([A-Za-z0-9]+)_")

# ===============================
# Robust CSV Reader / Writer
# ===============================
def read_csv_robust(path: Path, encoding: str = ENCODING) -> pd.DataFrame:
    """
    Robust CSV reader with multiple fallbacks. Returns DataFrame with string dtype
    and keep_default_na=False (so blanks stay as empty strings).
    """
    base = dict(dtype=str, keep_default_na=False)
    attempts = [
        dict(encoding=encoding),
        dict(encoding=encoding, engine="python", sep=None),
        dict(encoding=encoding, engine="python", sep=None, on_bad_lines="skip"),
        dict(encoding=encoding, engine="python", sep=",", quotechar='"', escapechar="\\", on_bad_lines="skip"),
    ]
    for enc in ("utf-16", "utf-16le", "latin1"):
        attempts.append(dict(encoding=enc, engine="python", sep=None, on_bad_lines="skip"))

    last_err = None
    for kw in attempts:
        try:
            return pd.read_csv(path, **base, **kw)
        except TypeError as te:
            # older pandas may not accept on_bad_lines; try older args
            if "on_bad_lines" in kw:
                kw2 = kw.copy()
                kw2.pop("on_bad_lines", None)
                kw2["error_bad_lines"] = False  # older pandas compat
                kw2["warn_bad_lines"] = True
                try:
                    return pd.read_csv(path, **base, **kw2)
                except Exception as e:
                    last_err = e
            else:
                last_err = te
        except Exception as e:
            last_err = e
    raise last_err

def _safe_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)

def safe_write_csv(df: pd.DataFrame, out_path: Path, encoding: str = ENCODING):
    """
    Attempts to save CSV, with fallback names if target busy. Creates OUT_DIR if missing.
    (This is a helper for CLI/batch usage only — UI wrappers do not call this.)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(out_path, index=False, encoding=encoding)
        return out_path
    except PermissionError:
        pass
    except OSError:
        spath = out_path.parent / f"{_safe_stem(out_path.stem)}{out_path.suffix}"
        try:
            df.to_csv(spath, index=False, encoding=encoding)
            return spath
        except Exception:
            out_path = spath

    base = _safe_stem(out_path.stem)
    suffix = out_path.suffix
    parent = out_path.parent
    n = 1
    while True:
        candidate = parent / f"{base}_{n}{suffix}"
        try:
            df.to_csv(candidate, index=False, encoding=encoding)
            return candidate
        except (PermissionError, OSError):
            n += 1

# ===============================
# Filename helpers
# ===============================
def extract_date_token(p: Path) -> Optional[str]:
    m = DATE_TOKEN_FINDER.search(p.name)
    return m.group(1) if m else None

def parse_token_to_date(tok: str) -> Optional[datetime]:
    if not tok:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", tok):
        fmts = ["%Y-%m-%d"]
    elif re.fullmatch(r"\d{2}-\d{2}-\d{4}", tok):
        fmts = ["%d-%m-%Y", "%m-%d-%Y"]
    elif re.fullmatch(r"\d{2}-\d{2}-\d{2}", tok):
        fmts = ["%d-%m-%y", "%m-%d-%y"]
    else:
        return None

    for fmt in fmts:
        try:
            d = datetime.strptime(tok, fmt)
            if "%y" in fmt and d.year < 2000:
                d = d.replace(year=d.year + 2000)
            return d
        except ValueError:
            continue
    return None

def extract_brand(p: Path) -> str:
    m = BRAND_RE.match(p.stem)
    return m.group(1).lower() if m else p.stem.lower()

# ===============================
# Dataframe helpers
# ===============================
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def _token(s: str) -> str:
    return re.sub(r"[\s_]+", "", s.lower())

def find_ci(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    tok = {_token(c): c for c in df.columns}
    for name in candidates:
        nlow = name.lower()
        if nlow in low: return low[nlow]
        ntok = _token(name)
        if ntok in tok: return tok[ntok]
    return None

def find_all_ci(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    found = []
    tok_map = {_token(c): c for c in df.columns}
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        n = cand.lower()
        if n in cols_lower:
            for c in df.columns:
                if c.lower() == n:
                    found.append(c)
        else:
            t = re.sub(r"[\s_]+", "", cand.lower())
            if t in tok_map:
                found.append(tok_map[t])
    # keep order and unique
    seen = set(); out = []
    for c in found:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def detect_lat_lon_optional(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
        return "LATITUDE", "LONGITUDE"
    low = {c.lower(): c for c in df.columns}
    lat = low.get("latitude") or low.get("lat")
    lon = low.get("longitude") or low.get("lon") or low.get("lng") or low.get("long")
    return lat, lon

def to_float(s):
    return pd.to_numeric(s, errors="coerce")

# ===============================
# Phone normalization & compare
# ===============================
def norm_phone(s: str) -> str:
    if s is None: return ""
    return re.sub(r"\D", "", str(s))

def phone_matches(a: str, b: str) -> bool:
    d1, d2 = norm_phone(a), norm_phone(b)
    if not d1 or not d2: return False
    if d1 == d2: return True
    return len(d1) >= 7 and len(d2) >= 7 and d1[-7:] == d2[-7:]

# ===============================
# Geo + Text normalization
# ===============================
def haversine_m(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return float("inf")
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlbd = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlbd/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def shared_bin_steps(all_lats: np.ndarray, meters: float) -> Tuple[float, float]:
    med_lat = float(np.nanmedian(all_lats)) if len(all_lats) else 0.0
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * max(math.cos(math.radians(med_lat)), 1e-6))
    return meters * deg_per_m_lat, meters * deg_per_m_lon

def assign_bins(df: pd.DataFrame, la: Optional[str], lo: Optional[str], lat_step: float, lon_step: float):
    if la and lo and la in df.columns and lo in df.columns:
        lat_vals = to_float(df[la]); lon_vals = to_float(df[lo])
        df["_lat_bin"] = np.where(lat_vals.notna(), np.floor(lat_vals / lat_step), np.nan).astype(np.float64)
        df["_lon_bin"] = np.where(lon_vals.notna(), np.floor(lon_vals / lon_step), np.nan).astype(np.float64)
    else:
        df["_lat_bin"] = np.nan
        df["_lon_bin"] = np.nan

def strip_accents(s: str) -> str:
    if s is None: return ""
    s = str(s)
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def norm_addr(s: str) -> str:
    if s is None: return ""
    t = strip_accents(str(s)).lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    subs = { r"\bst\.?\b":"street", r"\brd\.?\b":"road", r"\bave\.?\b":"avenue",
             r"\bblvd\.?\b":"boulevard", r"\bhwy\.?\b":"highway", r"\bdr\.?\b":"drive",
             r"\bln\.?\b":"lane", r"\bct\.?\b":"court" }
    for pat,rep in subs.items(): t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t)

def norm_city(s: str) -> str:
    if s is None: return ""
    t = strip_accents(str(s)).lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"^(st|st\.)\s+", "saint ", t)
    t = re.sub(r"^(west|east|north|south|w|e|n|s)\s+", "", t)
    t = re.sub(r"\s+(west|east|north|south|w|e|n|s)$", "", t)
    return re.sub(r"\s+", " ", t)

def norm_state(s: str) -> str:
    return "" if s is None else str(s).strip().upper()

def norm_zip(s: str) -> str:
    return "" if s is None else re.sub(r"\D", "", str(s))

def addr_key(row: pd.Series, a: Optional[str], c: Optional[str], s: Optional[str], z: Optional[str]) -> str:
    parts = []
    if a: parts.append(norm_addr(row.get(a, "")))
    if c: parts.append(norm_city(row.get(c, "")))
    if s: parts.append(norm_state(row.get(s, "")))
    if z: parts.append(norm_zip(row.get(z, "")))
    return " | ".join([p for p in parts if p])

def addr_line_key(row: pd.Series, a_col: Optional[str]) -> str:
    """Street line only, normalized; ignores city/state/zip."""
    return norm_addr(row.get(a_col, "")) if a_col else ""

def address_similarity(a: str, b: str) -> float:
    # De-emphasize separators
    a = a.replace("|", " ")
    b = b.replace("|", " ")
    sm = SequenceMatcher(None, a, b).ratio()
    ta = set(a.split()); tb = set(b.split())
    if not ta or not tb:
        return sm
    inter = len(ta & tb)
    cont_a = inter / len(ta)  # how much of A is in B
    cont_b = inter / len(tb)  # how much of B is in A
    jacc  = inter / len(ta | tb)
    ts = max(cont_a, cont_b, jacc)
    return max(sm, ts)

# ===============================
# Store ID normalization (numeric + GUID)
# ===============================
def normalize_id_value(v) -> Optional[str]:
    if v is None: return None
    s = str(v).strip()
    if not s or s.lower() in {"nan","none","null"}: return None
    s2 = re.sub(r"[^0-9a-zA-Z]", "", s).lower()  # remove hyphens/braces/spaces
    return s2 or None

def id_key_set(v) -> Set[str]:
    k = normalize_id_value(v)
    if not k: return set()
    keys = {k}
    if k.isdigit():
        keys.add(k.lstrip("0") or "0")  # bridge "00123" vs "123"
    return keys

# ===============================
# Schema mapping NEW → OLD (updated)
# ===============================
def get_old_id_col(old_cols_ordered: List[str]) -> Optional[str]:
    for col in old_cols_ordered:
        cl = col.lower()
        ct = re.sub(r"[\s_]+", "", cl)
        for cand in ID_CANDS:
            if cl == cand.lower() or ct == re.sub(r"[\s_]+", "", cand.lower()):
                return col
    return None

def map_new_to_old_schema(new_df: pd.DataFrame, old_cols_ordered: List[str]) -> pd.DataFrame:
    """
    Rename NEW columns to OLD names case-insensitively; add blanks for missing.
    Additionally: if OLD has multiple name-like columns and NEW has one, fill missing old name cols.
    Special handling for 'dba' column:
      - If OLD has a `dba` column and NEW does NOT have any dba-like column, do NOT populate OLD.dba
        from a generic NEW.name column. This preserves original DBA values in OLD.
      - If NEW contains an explicit dba-like column, it will be used to populate OLD.dba when empty.
    Also: map address1/address_1 -> old address column when possible.
    """
    new_lower = {c.lower(): c for c in new_df.columns}
    new_token = {re.sub(r"[\s_]+", "", c.lower()): c for c in new_df.columns}
    rename: Dict[str, str] = {}
    for old_col in old_cols_ordered:
        nm = new_lower.get(old_col.lower()) or new_token.get(re.sub(r"[\s_]+", "", old_col.lower()))
        if nm and nm != old_col:
            rename[nm] = old_col
    new2 = new_df.rename(columns=rename).copy()
    for c in old_cols_ordered:
        if c not in new2.columns:
            new2[c] = ""

    # ID aliasing/fill
    old_id_col = get_old_id_col(old_cols_ordered)
    if old_id_col and new2[old_id_col].eq("").all():
        new_id_col = find_ci(new_df, ID_CANDS)
        if new_id_col:
            new2[old_id_col] = new_df[new_id_col].fillna("").astype(str)

    # ADDRESS aliasing/fill: if OLD has an 'address' column but it's empty in projection,
    # try to fill from common address-like columns in NEW (address1/address_1/etc).
    old_addr_col = None
    addr_tokens = {re.sub(r"[\s_]+", "", c.lower()) for c in ADDR_CANDS}
    for col in old_cols_ordered:
        if re.sub(r"[\s_]+", "", col.lower()) in addr_tokens:
            old_addr_col = col
            break
    if old_addr_col and new2[old_addr_col].eq("").all():
        # find any address-like column in the original NEW
        new_addr_col = find_ci(new_df, ADDR_CANDS)
        if not new_addr_col:
            new_addr_cols = find_all_ci(new_df, ADDR_CANDS)
            new_addr_col = new_addr_cols[0] if new_addr_cols else None
        if new_addr_col:
            new2[old_addr_col] = new_df[new_addr_col].fillna("").astype(str)

    # NAME-like filling with special dba handling
    old_name_cols = [c for c in old_cols_ordered if re.sub(r"[\s_]+", "", c.lower()) in {re.sub(r"[\s_]+", "", n) for n in NAME_CANDS}]
    # find a generic new name (first best) and also list all name-like cols in new
    new_name_col_generic = find_ci(new_df, NAME_CANDS)
    new_name_cols_all = find_all_ci(new_df, NAME_CANDS)

    for col in old_name_cols:
        # token for this old name column (e.g., 'dba', 'storename' -> tokens without underscores/spaces)
        old_tok = re.sub(r"[\s_]+", "", col.lower())

        # If the old column is 'dba', only populate it from NEW if NEW has an explicit dba column.
        if old_tok == "dba":
            dba_in_new = None
            for nc in new_name_cols_all:
                if re.sub(r"[\s_]+", "", nc.lower()) == "dba":
                    dba_in_new = nc
                    break
            if dba_in_new:
                if col in new2 and new2[col].eq("").all():
                    new2[col] = new_df[dba_in_new].fillna("").astype(str)
            else:
                # intentionally leave OLD.dba alone (do not populate from generic name)
                continue
        else:
            # If new has a column matching same token (e.g., old 'store_name' and new 'store_name'), use that
            same_token_new = None
            for nc in new_name_cols_all:
                if re.sub(r"[\s_]+", "", nc.lower()) == old_tok:
                    same_token_new = nc
                    break
            if same_token_new:
                if col in new2 and new2[col].eq("").all():
                    new2[col] = new_df[same_token_new].fillna("").astype(str)
            else:
                # fallback: if generic new name exists (like 'name' or 'store_name'), use it for non-dba old name cols
                if new_name_col_generic and col in new2 and new2[col].eq("").all():
                    new2[col] = new_df[new_name_col_generic].fillna("").astype(str)

    return new2[old_cols_ordered]

# ===============================
# Compare a single OLD/NEW pair
# ===============================
def compare_pair(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Core compare logic. Inputs: old_df, new_df (pandas DataFrames).
    Returns: DataFrame with columns: ['flag','changed_columns', ...old schema...]
    Flags: 'matched', 'new', 'closed'
    """
    schema = list(old_df.columns)

    # locate columns (geo & phone optional)
    lat_o, lon_o = detect_lat_lon_optional(old_df)
    lat_n, lon_n = detect_lat_lon_optional(new_df)
    id_o  = find_ci(old_df, ID_CANDS)
    id_n  = find_ci(new_df, ID_CANDS)

    # address: try primary find_ci, otherwise pick the first address-like column found
    a_o = find_ci(old_df, ADDR_CANDS)
    if not a_o:
        a_o_list = find_all_ci(old_df, ADDR_CANDS)
        a_o = a_o_list[0] if a_o_list else None
    a_n = find_ci(new_df, ADDR_CANDS)
    if not a_n:
        a_n_list = find_all_ci(new_df, ADDR_CANDS)
        a_n = a_n_list[0] if a_n_list else None

    c_o, s_o, z_o = (find_ci(old_df, CITY_CANDS), find_ci(old_df, STATE_CANDS), find_ci(old_df, ZIP_CANDS))
    c_n, s_n, z_n = (find_ci(new_df, CITY_CANDS), find_ci(new_df, STATE_CANDS), find_ci(new_df, ZIP_CANDS))
    p_o = find_ci(old_df, PHONE_CANDS)
    p_n = find_ci(new_df, PHONE_CANDS)

    # Also get all name-like columns in old/new
    old_name_cols = find_all_ci(old_df, NAME_CANDS)
    new_name_cols = find_all_ci(new_df, NAME_CANDS)

    # copies & types
    old_df = old_df.copy(); new_df = new_df.copy()
    if lat_o and lon_o:
        old_df[lat_o] = to_float(old_df[lat_o]); old_df[lon_o] = to_float(old_df[lon_o])
    if lat_n and lon_n:
        new_df[lat_n] = to_float(new_df[lat_n]); new_df[lon_n] = to_float(new_df[lon_n])

    # binning (for geo corroboration only)
    lat_arrays = []
    if lat_o: lat_arrays.append(old_df[lat_o].dropna().to_numpy())
    if lat_n: lat_arrays.append(new_df[lat_n].dropna().to_numpy())
    all_lats = np.concatenate(lat_arrays) if lat_arrays else np.array([], dtype=float)
    lat_step, lon_step = shared_bin_steps(all_lats, MAX_METERS if len(all_lats) else 1.0)
    assign_bins(old_df, lat_o, lon_o, lat_step, lon_step)
    assign_bins(new_df, lat_n, lon_n, lat_step, lon_step)

    # project NEW to OLD schema (for updates & 'new' rows)
    new_proj = map_new_to_old_schema(new_df, schema)

    used_new: Set[int] = set()
    pair: Dict[int, int] = {}

    # ==== FAST PASS: exact normalized composite address equality ====
    try:
        if a_o:
            old_comps = {}
            for i in range(len(old_df)):
                old_comps[i] = addr_key(old_df.loc[i], a_o, c_o, s_o, z_o).strip().lower()
            new_comps = {}
            for j in range(len(new_proj)):
                new_comps[j] = addr_key(new_proj.loc[j], a_n, c_n, s_n, z_n).strip().lower()

            new_map = {}
            for j, comp in new_comps.items():
                if comp:
                    new_map.setdefault(comp, []).append(j)

            for i, comp in old_comps.items():
                if not comp:
                    continue
                if comp in new_map:
                    found = None
                    for j in new_map[comp]:
                        if j not in used_new:
                            found = j; break
                    if found is not None:
                        pair[i] = found
                        used_new.add(found)
    except Exception:
        # safe fallback if something unexpected in columns
        pass

    # ---- helpers ----
    def dist_ij(i, j) -> float:
        if not (lat_o and lon_o and lat_n and lon_n): return float("inf")
        return haversine_m(old_df.at[i, lat_o], old_df.at[i, lon_o],
                           new_df.at[j, lat_n], new_df.at[j, lon_n])

    def build_indexes():
        by_zip: Dict[str, List[int]] = {}
        by_city_state: Dict[Tuple[str,str], List[int]] = {}
        by_state: Dict[str, List[int]] = {}
        all_idx: List[int] = []
        for j in range(len(new_df)):
            if j in used_new: continue
            all_idx.append(j)
            if z_n:
                z = norm_zip(new_df.at[j, z_n])
                if z: by_zip.setdefault(z, []).append(j)
            if c_n or s_n:
                ck = norm_city(new_df.at[j, c_n]) if c_n else ""
                sk = norm_state(new_df.at[j, s_n]) if s_n else ""
                if ck or sk:
                    by_city_state.setdefault((ck, sk), []).append(j)
            if s_n:
                sk = norm_state(new_df.at[j, s_n])
                if sk:
                    by_state.setdefault(sk, []).append(j)
        return by_zip, by_city_state, by_state, all_idx

    by_zip, by_city_state, by_state, all_idx = build_indexes()

    # =========================
    # A) ADDRESS-FIRST (fuzzy + street-line containment)
    # =========================
    for i in range(len(old_df)):
        if i in pair: continue

        # candidate pool: same ZIP > same CITY+STATE > same STATE > all
        cand_pool: List[int] = []
        z = norm_zip(old_df.at[i, z_o]) if z_o else ""
        if z and z in by_zip:
            cand_pool = by_zip[z].copy()
        else:
            ck = norm_city(old_df.at[i, c_o]) if c_o else ""
            sk = norm_state(old_df.at[i, s_o]) if s_o else ""
            if (ck or sk) and (ck, sk) in by_city_state:
                cand_pool = by_city_state[(ck, sk)].copy()
            elif sk and sk in by_state:
                cand_pool = by_state[sk].copy()
            else:
                cand_pool = all_idx.copy()

        if not cand_pool:
            continue

        addr_old_comp = addr_key(old_df.loc[i], a_o, c_o, s_o, z_o)
        addr_old_line = addr_line_key(old_df.loc[i], a_o)
        phone_old = old_df.at[i, p_o] if p_o else ""

        best = (0.0, 0.0, 0, float("-inf"), None)  # (a_sim_comp, line_sim, phone_eq, -distance, j)
        for j in cand_pool:
            if j in used_new:
                continue
            addr_new_comp = addr_key(new_df.loc[j], a_n, c_n, s_n, z_n)
            addr_new_line = addr_line_key(new_df.loc[j], a_n)

            a_sim_comp = address_similarity(addr_old_comp, addr_new_comp)

            # street-line containment score
            ta = set(addr_old_line.split()); tb = set(addr_new_line.split())
            inter = len(ta & tb)
            line_sim = 0.0 if not ta or not tb else max(inter/len(ta), inter/len(tb), inter/len(ta|tb))

            if p_o and p_n:
                phone_eq = 1 if phone_matches(phone_old, new_df.at[j, p_n]) else 0
            else:
                phone_eq = 0

            d = dist_ij(i, j)
            cand = (a_sim_comp, line_sim, phone_eq, -d, j)
            if cand > best:
                best = cand

        a_sim_comp, line_sim, phone_eq, neg_d, jbest = best
        if jbest is None:
            continue
        d = -neg_d

        # containment: old street tokens ⊆ new OR new ⊆ old
        old_tokens = set(addr_old_line.split())
        new_tokens = set(addr_line_key(new_df.loc[jbest], a_n).split())
        subset_line = (old_tokens <= new_tokens) or (new_tokens <= old_tokens)

        accept = False
        # 1) Street-line containment or near-perfect line similarity
        if subset_line or line_sim >= 0.98:
            accept = True
        # 2) Composite address very close
        elif a_sim_comp >= ADDR_STRICT:
            accept = True
        # 3) Decent text + phone OR geo corroboration
        elif max(a_sim_comp, line_sim) >= ADDR_LOOSE and (phone_eq == 1 or d <= MAX_METERS):
            accept = True
        # 4) Strong corroboration: phone + geo
        elif phone_eq == 1 and d <= MAX_METERS:
            accept = True

        if accept and jbest not in used_new:
            pair[i] = jbest
            used_new.add(jbest)
            # remove from indexes for uniqueness
            if z_n:
                zz = norm_zip(new_df.at[jbest, z_n])
                if zz in by_zip and jbest in by_zip[zz]: by_zip[zz].remove(jbest)
            if c_n or s_n:
                ck2 = norm_city(new_df.at[jbest, c_n]) if c_n else ""
                sk2 = norm_state(new_df.at[jbest, s_n]) if s_n else ""
                if (ck2 or sk2) and (ck2, sk2) in by_city_state and jbest in by_city_state[(ck2, sk2)]:
                    by_city_state[(ck2, sk2)].remove(jbest)
            if s_n:
                sk2 = norm_state(new_df.at[jbest, s_n])
                if sk2 in by_state and jbest in by_state[sk2]: by_state[sk2].remove(jbest)
            if jbest in all_idx: all_idx.remove(jbest)

    # =========================
    # B) STORE ID (fallback / tie-break)
    # =========================
    if id_o and id_n:
        new_by_key: Dict[str, List[int]] = {}
        for j, v in new_df[id_n].items():
            if j in used_new: continue
            for k in id_key_set(v):
                new_by_key.setdefault(k, []).append(j)

        for i, v in old_df[id_o].items():
            if i in pair: continue
            keys = id_key_set(v)
            if not keys:
                continue
            cand_js: List[int] = []
            for k in keys:
                cand_js.extend(new_by_key.get(k, []))
            if not cand_js:
                continue

            # pick the best by address similarity then phone then distance
            addr_old = addr_key(old_df.loc[i], a_o, c_o, s_o, z_o)
            phone_old = old_df.at[i, p_o] if p_o else ""
            best = (0.0, 0, float("-inf"), None)  # (addr_sim, phone_eq, -distance, j)
            for j in cand_js:
                if j in used_new: continue
                addr_new = addr_key(new_df.loc[j], a_n, c_n, s_n, z_n)
                a_sim = address_similarity(addr_old, addr_new)
                if p_o and p_n:
                    phone_eq = 1 if phone_matches(phone_old, new_df.at[j, p_n]) else 0
                else:
                    phone_eq = 0
                d = dist_ij(i, j)
                cand = (a_sim, phone_eq, -d, j)
                if cand > best:
                    best = cand

            jbest = best[-1]
            if jbest is not None and jbest not in used_new:
                pair[i] = jbest
                used_new.add(jbest)

    # --- partition ---
    matched_old = sorted(pair.keys())
    matched_new = sorted(set(pair.values()))
    closed_old  = sorted(set(range(len(old_df))) - set(matched_old))
    new_only    = sorted(set(range(len(new_df))) - set(matched_new))

    # --- do not overwrite city/state/zip on matched rows ---
    do_not_update_cols: Set[str] = set()
    for colset in (CITY_CANDS, STATE_CANDS, ZIP_CANDS):
        col = find_ci(old_df, colset)
        if col:
            do_not_update_cols.add(col)
    for c in old_df.columns:
        if c.lower() in DO_NOT_OVERWRITE:
            do_not_update_cols.add(c)

    # --- merge logic (OLD base, NEW overwrites on change, EXCEPT excluded cols) ---
    def merge_old_with_new(old_row: pd.Series, new_row_proj: pd.Series):
        merged = {"flag": "matched"}
        changed: List[str] = []
        for col in schema:
            old_val = "" if pd.isna(old_row.get(col)) else str(old_row.get(col))
            new_val = "" if pd.isna(new_row_proj.get(col)) else str(new_row_proj.get(col))
            if col in do_not_update_cols:
                merged[col] = old_val
            elif new_val != "" and new_val != old_val:
                merged[col] = new_val; changed.append(col)
            else:
                merged[col] = old_val
        return merged, changed

    # --- output rows ---
    out_rows: List[Dict[str, str]] = []
    for i in matched_old:
        j = pair[i]
        merged_dict, changed_cols = merge_old_with_new(old_df.loc[i], new_proj.loc[j])
        merged_dict["changed_columns"] = ", ".join(changed_cols)
        out_rows.append(merged_dict)

    for i in closed_old:
        rec = {"flag": "closed", "changed_columns": ""}
        for col in schema:
            rec[col] = old_df.at[i, col]
        out_rows.append(rec)

    for j in new_only:
        rec = {"flag": "new", "changed_columns": ""}
        for col in schema:
            rec[col] = new_proj.at[j, col] if col in new_proj.columns else ""
        out_rows.append(rec)

    out = pd.DataFrame(out_rows)
    # ensure columns exist before selecting
    selected = ["flag", "changed_columns"] + [c for c in schema if c in out.columns]
    return out[selected]

# ===============================
# Pairing across many files (by brand)
# ===============================
def list_csvs(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.csv"))

def build_index(files: List[Path]):
    items = []
    for p in files:
        tok = extract_date_token(p)
        dt  = parse_token_to_date(tok) if tok else None
        items.append({"path": p, "brand": extract_brand(p), "token": tok, "date": dt})
    return items

def pair_files_by_brand(old_files: List[Path], new_files: List[Path]) -> List[Tuple[Path, Path]]:
    old_items = build_index(old_files)
    new_items = build_index(new_files)

    # group by brand
    olds_by_brand: Dict[str, List[Dict]] = {}
    for it in old_items:
        olds_by_brand.setdefault(it["brand"], []).append(it)
    news_by_brand: Dict[str, List[Dict]] = {}
    for it in new_items:
        news_by_brand.setdefault(it["brand"], []).append(it)

    pairs: List[Tuple[Path, Path]] = []
    for brand, news in news_by_brand.items():
        olds = sorted(olds_by_brand.get(brand, []), key=lambda x: (x["date"] or datetime.min, x["path"].stat().st_mtime))
        if not olds:
            continue

        if PAIRING_STRATEGY == "same-date":
            old_by_token: Dict[str, List[Dict]] = {}
            for it in olds:
                if it["token"]:
                    old_by_token.setdefault(it["token"], []).append(it)
            for nw in news:
                tok = nw["token"]
                if tok and tok in old_by_token:
                    op = sorted(old_by_token[tok], key=lambda x: x["path"].stat().st_mtime)[-1]
                    pairs.append((op["path"], nw["path"]))
        else:
            for nw in news:
                dtn = nw["date"]
                if dtn:
                    candidates = [it for it in olds if (it["date"] and it["date"] <= dtn)]
                    op = candidates[-1] if candidates else olds[-1]
                else:
                    op = olds[-1]
                pairs.append((op["path"], nw["path"]))

    # dedup exact pairs
    seen: Set[Tuple[str, str]] = set()
    dedup: List[Tuple[Path, Path]] = []
    for op, np_ in pairs:
        key = (str(op), str(np_))
        if key not in seen:
            seen.add(key); dedup.append((op, np_))
    return dedup

# ===============================
# Batch run (optional CLI)
# ===============================
def run_batch(save_results: bool = True):
    """
    Optional batch mode: reads CSVs from OLD_DIR and NEW_DIR, pairs by brand and compares.
    If save_results is True, writes outputs to OUT_DIR using safe_write_csv.
    """
    old_files = list_csvs(OLD_DIR)
    new_files = list_csvs(NEW_DIR)
    if not old_files: raise FileNotFoundError(f"No CSVs found in {OLD_DIR}")
    if not new_files: raise FileNotFoundError(f"No CSVs found in {NEW_DIR}")

    pairs = pair_files_by_brand(old_files, new_files)
    if not pairs:
        print("⚠️  No brand-matched pairs found with current strategy.")
        return

    print(f"🔎 Found {len(pairs)} pair(s) ({PAIRING_STRATEGY}) by brand.")
    results = []
    for old_path, new_path in pairs:
        print(f"• {extract_brand(old_path)} | OLD: {old_path.name}  ⇢  NEW: {new_path.name}")

        old_df = normalize_headers(read_csv_robust(old_path))
        new_df = normalize_headers(read_csv_robust(new_path))

        out_df = compare_pair(old_df, new_df)
        results.append((old_path, new_path, out_df))

        if save_results:
            out_path = OUT_DIR / f"{new_path.stem}_compared.csv"
            safe_write_csv(out_df, out_path)

    return results

# ===============================
# Streamlit / Upload wrappers (in-memory)
# ===============================
def compare_pair_df(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Public wrapper for UI. Accepts two DataFrames (old_df, new_df).
    Returns compared DataFrame. No file writes, purely in-memory.
    """
    # ensure strings to avoid dtype surprises
    old_df = normalize_headers(old_df.copy().astype(object).where(pd.notna(old_df), ""))
    new_df = normalize_headers(new_df.copy().astype(object).where(pd.notna(new_df), ""))
    return compare_pair(old_df, new_df)

def compare_uploaded_files(old_file, new_file) -> pd.DataFrame:
    """
    Wrapper for UI:
    Accepts uploaded file-like objects (Streamlit) OR filesystem path (Path/str).
    Returns the compared DataFrame (in-memory).
    """
    if hasattr(old_file, "read"):
        old_df = pd.read_csv(old_file, dtype=str, keep_default_na=False, encoding=ENCODING)
    else:
        old_df = normalize_headers(read_csv_robust(Path(old_file)))

    if hasattr(new_file, "read"):
        new_df = pd.read_csv(new_file, dtype=str, keep_default_na=False, encoding=ENCODING)
    else:
        new_df = normalize_headers(read_csv_robust(Path(new_file)))

    return compare_pair_df(old_df, new_df)

def compare_files_paths(old_path: str, new_path: str) -> pd.DataFrame:
    """
    Simple helper to compare two filesystem CSV paths. Returns DataFrame in-memory.
    """
    old_df = normalize_headers(read_csv_robust(Path(old_path)))
    new_df = normalize_headers(read_csv_robust(Path(new_path)))
    return compare_pair_df(old_df, new_df)

# End of module - nothing executes on import.
