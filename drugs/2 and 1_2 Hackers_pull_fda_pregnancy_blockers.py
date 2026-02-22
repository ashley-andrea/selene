import csv, json, time, requests
from pathlib import Path

OUTDIR = Path("fda_pregnancy_blockers")
OUTDIR.mkdir(exist_ok=True)
BASE = "https://api.fda.gov/drug/ndc.json"

# Emergency contraceptives + medical abortion drugs
# Each entry: (ingredient, route, dosage_form)
# route/dosage_form can be None to skip that filter
QUERIES = [
    # Morning-after / emergency contraceptives
    ("LEVONORGESTREL",    "ORAL",    "TABLET"),
    ("ULIPRISTAL ACETATE","ORAL",    "TABLET"),
    # Medical abortion (mifepristone is always oral tablet)
    ("MIFEPRISTONE",      "ORAL",    "TABLET"),
    # Misoprostol is used orally, buccally, sublingually, vaginally
    ("MISOPROSTOL",       "ORAL",    None),
    ("MISOPROSTOL",       "BUCCAL",  None),
    ("MISOPROSTOL",       "VAGINAL", None),
    ("MISOPROSTOL",       "SUBLINGUAL", None),
]

def fetch_all(search: str, limit: int = 100, max_pages: int = 200):
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

def flatten_active_ingredients(ai_list):
    if not ai_list:
        return ""
    return "; ".join(
        f'{ai.get("name","")} ({ai.get("strength","")})'.strip()
        for ai in ai_list
    )

def build_query(ingredient, route, dosage_form):
    q = f'active_ingredients.name:"{ingredient}"'
    if route:
        q += f' AND route:"{route}"'
    if dosage_form:
        q += f' AND dosage_form:"{dosage_form}"'
    return q

def main():
    by_ndc = {}

    for ingredient, route, dosage_form in QUERIES:
        q = build_query(ingredient, route, dosage_form)
        print(f"Querying: {q}")
        rows = fetch_all(q)
        print(f"  -> {len(rows)} results")
        for r in rows:
            ndc = r.get("product_ndc")
            if ndc:
                by_ndc[ndc] = r

    rows = list(by_ndc.values())

    raw_path = OUTDIR / "pregnancy_blockers_raw.json"
    raw_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = OUTDIR / "pregnancy_blockers_flat.csv"
    fieldnames = [
        "product_ndc", "brand_name", "generic_name",
        "route", "dosage_form", "active_ingredients",
        "marketing_category", "labeler_name",
        "start_marketing_date", "end_marketing_date",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "product_ndc":           r.get("product_ndc", ""),
                "brand_name":            r.get("brand_name", ""),
                "generic_name":          r.get("generic_name", ""),
                "route":                 r.get("route", ""),
                "dosage_form":           r.get("dosage_form", ""),
                "active_ingredients":    flatten_active_ingredients(r.get("active_ingredients", [])),
                "marketing_category":    r.get("marketing_category", ""),
                "labeler_name":          r.get("labeler_name", ""),
                "start_marketing_date":  r.get("start_marketing_date", ""),
                "end_marketing_date":    r.get("end_marketing_date", ""),
            })

    print("\nDone.")
    print("Records:", len(rows))
    print("Wrote:", raw_path, "and", csv_path)

if __name__ == "__main__":
    main()
