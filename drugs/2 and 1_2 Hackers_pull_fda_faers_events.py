import csv, json, time, requests
from pathlib import Path

OUTDIR = Path("fda_faers_events")
OUTDIR.mkdir(exist_ok=True)
BASE = "https://api.fda.gov/drug/event.json"

# Generic names to search in FAERS reports
# FAERS uses medicinalproduct (as-reported name), we use openfda.generic_name for consistency
DRUG_NAMES = [
    "ethinyl estradiol",
    "norethindrone",
    "levonorgestrel",
    "norgestimate",
    "desogestrel",
    "drospirenone",
    "norgestrel",
    "ethynodiol diacetate",
    "norethindrone acetate",
    "ulipristal acetate",
    "mifepristone",
    "misoprostol",
]

# Cap per drug — FAERS can have 100k+ reports for common drugs; keep it manageable
MAX_PAGES_PER_DRUG = 20  # 20 * 100 = 2000 reports max per drug

def fetch_all(search: str, limit: int = 100, max_pages: int = MAX_PAGES_PER_DRUG):
    all_rows = []
    skip = 0
    for _ in range(max_pages):
        params = {"search": search, "limit": limit, "skip": skip}
        r = requests.get(BASE, params=params, timeout=30)
        if r.status_code == 404:
            break
        r.raise_for_status()
        data = r.json()
        rows = data.get("results", [])
        all_rows.extend(rows)
        if len(rows) < limit:
            break
        skip += limit
        time.sleep(0.25)
    return all_rows

def get_reactions(event):
    """Extract all MedDRA reaction preferred terms from an event."""
    reactions = event.get("patient", {}).get("reaction", [])
    return " | ".join(r.get("reactionmeddrapt", "") for r in reactions if r.get("reactionmeddrapt"))

def get_drugs(event):
    """Extract suspect drug names from the event."""
    drugs = event.get("patient", {}).get("drug", [])
    # characterization: 1=suspect, 2=concomitant, 3=interacting
    suspect = [d for d in drugs if str(d.get("drugcharacterization", "")) == "1"]
    names = []
    for d in suspect:
        name = d.get("openfda", {}).get("generic_name", [])
        if name:
            names.extend(name)
        else:
            names.append(d.get("medicinalproduct", ""))
    return " | ".join(filter(None, names))

def get_outcomes(event):
    """Seriousness flags — tells us how bad the adverse event was."""
    flags = {
        "seriousnesscongenitalanomali": "congenital_anomaly",
        "seriousnessdeath":             "death",
        "seriousnessdisabling":         "disabling",
        "seriousnesshospitalization":   "hospitalization",
        "seriousnesslifethreatening":   "life_threatening",
        "seriousnessother":             "other_serious",
    }
    active = [label for key, label in flags.items() if event.get(key) == "1"]
    return " | ".join(active) if active else ""

def main():
    by_report_id = {}

    for drug in DRUG_NAMES:
        q = f'patient.drug.openfda.generic_name:"{drug}"'
        print(f"Querying FAERS: {drug}")
        rows = fetch_all(q)
        print(f"  -> {len(rows)} reports")
        for r in rows:
            rid = r.get("safetyreportid")
            if rid:
                by_report_id[rid] = r

    records = list(by_report_id.values())

    raw_path = OUTDIR / "faers_events_raw.json"
    raw_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Flat CSV: one row per event ---
    csv_path = OUTDIR / "faers_events_flat.csv"
    fieldnames = [
        "safetyreportid",
        "receivedate",          # when FDA received the report
        "serious",              # 1=serious, 2=not serious
        "outcomes",             # which seriousness flags were set
        "patient_sex",          # 1=male, 2=female
        "patient_age_group",    # FDA age group code (1=neonate … 6=elderly)
        "suspect_drugs",        # generic names of suspect drugs
        "reactions",            # MedDRA reaction terms (pipe-separated)
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            patient = r.get("patient", {})
            w.writerow({
                "safetyreportid":  r.get("safetyreportid", ""),
                "receivedate":     r.get("receivedate", ""),
                "serious":         r.get("serious", ""),
                "outcomes":        get_outcomes(r),
                "patient_sex":     patient.get("patientsex", ""),
                "patient_age_group": patient.get("patientagegroup", ""),
                "suspect_drugs":   get_drugs(r),
                "reactions":       get_reactions(r),
            })

    # --- Reaction-exploded CSV: one row per reaction per event (better for analysis) ---
    reactions_path = OUTDIR / "faers_reactions_exploded.csv"
    reaction_fields = [
        "safetyreportid", "receivedate", "serious",
        "patient_sex", "patient_age_group",
        "suspect_drugs", "reaction_term",
    ]
    with reactions_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=reaction_fields)
        w.writeheader()
        for r in records:
            patient = r.get("patient", {})
            base = {
                "safetyreportid":    r.get("safetyreportid", ""),
                "receivedate":       r.get("receivedate", ""),
                "serious":           r.get("serious", ""),
                "patient_sex":       patient.get("patientsex", ""),
                "patient_age_group": patient.get("patientagegroup", ""),
                "suspect_drugs":     get_drugs(r),
            }
            reactions = patient.get("reaction", [])
            if reactions:
                for rx in reactions:
                    term = rx.get("reactionmeddrapt", "")
                    if term:
                        w.writerow({**base, "reaction_term": term})
            else:
                w.writerow({**base, "reaction_term": ""})

    print("\nDone.")
    print("Unique reports:", len(records))
    print("Wrote:")
    print(" ", raw_path)
    print(" ", csv_path)
    print(" ", reactions_path)

if __name__ == "__main__":
    main()
