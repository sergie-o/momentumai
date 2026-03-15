"""
yard_parser.py
──────────────
Reads a yard management Excel export (copy-pasted from the yard app) and
extracts pallet inventory counts broken down by type: wooden, plastic, mixed, other.

The file has a multi-row-per-shipment layout where each shipment block starts
with a PS-XXXX Load ID in column A, and pallet description text appears in the
rightmost column in patterns like:
  • "Pallet Storage 330 Wooden"
  • "Pallet Storage 300 Plastic"
  • "Pallet Storage Plastic & Wood"
  • "BTS2 Storage TSO 30 Pallets DS"
"""

import re
import io
import pandas as pd

# ── Regex patterns ────────────────────────────────────────────────────────────

# Mixed must come BEFORE wooden/plastic so it is not partially matched
_RE_MIXED   = re.compile(
    r'(\d+)\s*(?:pallets?)?\s*(?:plastic\s*[&and]+\s*wood|wood\s*[&and]+\s*plastic)'
    r'|(?:plastic\s*[&and]+\s*wood|wood\s*[&and]+\s*plastic).*?(\d+)',
    re.IGNORECASE,
)
_RE_WOODEN  = re.compile(r'(\d+)\s*(?:pallets?)?\s*wood|wood.*?(\d+)\s*pallets?', re.IGNORECASE)
_RE_PLASTIC = re.compile(r'(\d+)\s*(?:pallets?)?\s*plastic|plastic.*?(\d+)\s*pallets?', re.IGNORECASE)
_RE_BTS     = re.compile(r'bts\w*.*?(\d+)\s*pallets?|(\d+)\s*pallets?.*?bts\w*', re.IGNORECASE)

# Row-context patterns
_RE_LOAD_ID = re.compile(r'^PS-\d+', re.IGNORECASE)
_RE_TIME    = re.compile(r'^\d+\s*days?$|^\d{1,3}:\d{2}:\d{2}$', re.IGNORECASE)
_RE_VEH_ID  = re.compile(r'^(?:VS|IBS|HH|KRA)\w*\d{4,}', re.IGNORECASE)
_RE_OWNER   = re.compile(r'ATSEU', re.IGNORECASE)
_RE_INBOUND = re.compile(r'^INBOUND$', re.IGNORECASE)


# ── Cell-level pallet parser ──────────────────────────────────────────────────

def _parse_pallet_cell(text: str):
    """
    Try to extract (count: int, ptype: str) from a single cell string.
    ptype is one of: 'wooden', 'plastic', 'mixed', 'other'.
    Returns None if the cell does not describe pallets.
    """
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t or t.lower() == 'nan':
        return None

    # Guard: must contain 'pallet' or 'bts' to be a pallet description
    if not re.search(r'pallet|bts', t, re.IGNORECASE):
        return None

    # 1. Mixed (plastic & wood) — check first
    m = _RE_MIXED.search(t)
    if m:
        n = int(m.group(1) or m.group(2))
        return n, 'mixed'

    # 2. Wooden
    m = _RE_WOODEN.search(t)
    if m:
        n = int(m.group(1) or m.group(2))
        return n, 'wooden'

    # 3. Plastic
    m = _RE_PLASTIC.search(t)
    if m:
        n = int(m.group(1) or m.group(2))
        return n, 'plastic'

    # 4. BTS / TSO / DS (special pallet type)
    m = _RE_BTS.search(t)
    if m:
        n = int(m.group(1) or m.group(2))
        return n, 'other'

    return None


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_yard_excel(file_bytes: bytes) -> dict:
    """
    Parse a yard management Excel export.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the uploaded .xlsx file.

    Returns
    -------
    dict with keys:
        wooden          int   – total wooden pallets
        plastic         int   – total plastic pallets
        mixed           int   – total mixed-type pallets
        other           int   – total other/BTS pallets
        total           int   – grand total across all types
        shipment_count  int   – distinct PS-XXX load IDs found
        rows            list  – list of dicts, one per pallet-bearing shipment
        error           str|None
    """
    result = dict(
        wooden=0, plastic=0, mixed=0, other=0,
        total=0, shipment_count=0, rows=[], error=None,
    )

    # ── Load workbook ─────────────────────────────────────────────────────────
    try:
        df = pd.read_excel(
            io.BytesIO(file_bytes),
            header=None,
            dtype=str,
            engine='openpyxl',
        )
    except Exception as e:
        result['error'] = f"Could not read file: {e}"
        return result

    # ── Scan rows ─────────────────────────────────────────────────────────────
    load_ids_seen = set()
    ctx = dict(load_id=None, time=None, vehicle_id=None, owner=None, visit=None)

    for _, row in df.iterrows():
        cells = []
        for c in row:
            val = str(c).strip() if pd.notna(c) else ''
            cells.append(val if val.lower() != 'nan' else '')

        for cell in cells:
            if not cell:
                continue

            # ── New shipment block ────────────────────────────────────────────
            if _RE_LOAD_ID.match(cell):
                ctx = dict(load_id=cell, time=None,
                           vehicle_id=None, owner=None, visit=None)
                load_ids_seen.add(cell)
                continue

            if ctx['load_id'] is None:
                continue   # skip header rows before first load ID

            # ── Context fields ────────────────────────────────────────────────
            if _RE_TIME.match(cell):
                ctx['time'] = cell
                continue
            if _RE_VEH_ID.match(cell):
                ctx['vehicle_id'] = cell
                continue
            if _RE_OWNER.search(cell):
                ctx['owner'] = cell
                continue
            if _RE_INBOUND.match(cell):
                ctx['visit'] = cell
                continue

            # ── Pallet description ────────────────────────────────────────────
            pinfo = _parse_pallet_cell(cell)
            if pinfo:
                count, ptype = pinfo
                result[ptype] = result.get(ptype, 0) + count
                result['total'] += count
                result['rows'].append({
                    'Load ID':      ctx['load_id'],
                    'Time In Yard': ctx['time']      or '—',
                    'Vehicle ID':   ctx['vehicle_id'] or '—',
                    'Owner':        ctx['owner']      or '—',
                    'Visit':        ctx['visit']      or '—',
                    'Pallets':      count,
                    'Type':         ptype.capitalize(),
                    'Description':  cell,
                })

    result['shipment_count'] = len(load_ids_seen)
    return result
