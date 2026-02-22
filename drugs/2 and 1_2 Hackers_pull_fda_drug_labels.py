import csv, json, time, requests
from pathlib import Path

OUTDIR = Path("fda_drug_labels")
OUTDIR.mkdir(exist_ok=True)
BASE = "https://api.fda.gov/drug/label.json"

# Same ingredient list as NDC scripts — keeps all datasets linkable
INGREDIENTS = [
    "ETHINYL ESTRADIOL",
    "NORETHINDRONE",
    "LEVONORGESTREL",
    "NORGESTIMATE",
    "DESOGESTREL",
    "DROSPIRENONE",
    "NORGESTREL",
    "ETHYNODIOL DIACETATE",
    "NORETHINDRONE ACETATE",
    "ULIPRISTAL ACETATE",
    "MIFEPRISTONE",
    "MISOPROSTOL",
]

# OpenFDA only indexes some progestins under openfda.generic_name (as combo products),
# NOT under openfda.substance_name — so we override the search field for these.
# Verified: substance_name:"NORGESTIMATE" → 404; generic_name:"NORGESTIMATE" → 46 results
GENERIC_NAME_INGREDIENTS = {"NORGESTIMATE", "DESOGESTREL", "DROSPIRENONE"}

# Text sections from the label we want to capture
# These are the fields most relevant to side effects + clinical context
LABEL_FIELDS = [
    "adverse_reactions",          # documented side effects
    "warnings",                   # safety warnings
    "warnings_and_cautions",      # additional cautions
    "boxed_warning",              # black-box warnings (most serious)
    "contraindications",          # who should NOT take this
    "drug_interactions",          # interactions with other drugs/substances
    "clinical_pharmacology",      # how the drug works in the body
    "indications_and_usage",      # what it's officially approved to treat
    "dosage_and_administration",  # dosing info
    "description",                # chemical / formulation description
]

def fetch_all(search: str, limit: int = 100, max_pages: int = 50):
    all_rows = []
    skip = 0
    for _ in range(max_pages):
        params = {"search": search, "limit": limit, "skip": skip}
        r = requests.get(BASE, params=params, timeout=30)
        if r.status_code == 404:
            break
        r.raise_for_status()
        rows = r.json().get("results", [])
        all_rows.extend(rows)
        if len(rows) < limit:
            break
        skip += limit
        time.sleep(0.2)
    return all_rows

def extract_text(label, field):
    """Label text fields are returned as lists — join into a single string."""
    val = label.get(field)
    if not val:
        return ""
    if isinstance(val, list):
        return " ".join(val).strip()
    return str(val).strip()

def extract_openfda(label, field):
    """openFDA nested fields are lists — join with pipe."""
    val = label.get("openfda", {}).get(field, [])
    if isinstance(val, list):
        return " | ".join(val)
    return str(val)

def main():
    by_id = {}

    for ing in INGREDIENTS:
        if ing in GENERIC_NAME_INGREDIENTS:
            q = f'openfda.generic_name:"{ing}"'
        else:
            q = f'openfda.substance_name:"{ing}"'
        print(f"Querying labels: {ing}")
        rows = fetch_all(q)
        print(f"  -> {len(rows)} results")
        for r in rows:
            # When searching via generic_name, OpenFDA often omits substance_name —
            # inject the queried ingredient so load_fda_labels() can index by it.
            if ing in GENERIC_NAME_INGREDIENTS:
                openfda = r.setdefault("openfda", {})
                if not openfda.get("substance_name"):
                    openfda["substance_name"] = [ing]
            # Use set_id as dedup key (one label doc per product version)
            uid = r.get("set_id") or r.get("id")
            if uid:
                by_id[uid] = r

    records = list(by_id.values())

    raw_path = OUTDIR / "drug_labels_raw.json"
    raw_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = OUTDIR / "drug_labels_flat.csv"
    fieldnames = [
        "set_id",
        "brand_name", "generic_name", "manufacturer_name",
        "product_ndc",          # links to NDC datasets
        "substance_name",
    ] + LABEL_FIELDS

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = {
                "set_id":            r.get("set_id", ""),
                "brand_name":        extract_openfda(r, "brand_name"),
                "generic_name":      extract_openfda(r, "generic_name"),
                "manufacturer_name": extract_openfda(r, "manufacturer_name"),
                "product_ndc":       extract_openfda(r, "product_ndc"),
                "substance_name":    extract_openfda(r, "substance_name"),
            }
            for field in LABEL_FIELDS:
                row[field] = extract_text(r, field)
            w.writerow(row)

    print("\nDone.")
    print("Records:", len(records))
    print("Wrote:", raw_path, "and", csv_path)

if __name__ == "__main__":
    main()
