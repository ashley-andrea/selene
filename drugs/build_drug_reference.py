"""
build_drug_reference.py
=======================
Builds the unified pill reference database from three sources:
  1. pills_master.csv        — hand-curated pharmacological ground truth
  2. drugsComTrain/Test_raw  — patient-reported side effects (21k oral BC reviews)
  3. FAERS reactions_exploded— real-world serious adverse event signals
  4. FDA drug labels          — formal contraindication/warning text

Output:
  drugs/output/pill_reference_db.csv
       One row per canonical pill formulation.
       Columns: identity + pharmacology + side-effect frequencies +
                serious-event counts + formal safety text

Usage:
    cd /Users/ashleyandrea/Documents/Projects/2-and-1-2-hackers
    python drugs/build_drug_reference.py
"""

import csv, collections, html, re, os, json
from pathlib import Path

DRUGS_DIR = Path(__file__).parent
OUT_DIR   = DRUGS_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  1.  CANONICAL PILL TAXONOMY
#      Each canonical formulation is identified by a combo_id.
#      brand_aliases: all names that appear in drugscom / FAERS for this combo.
# ─────────────────────────────────────────────────────────────────────────────
PILL_MASTER = [
    # ── Combined monophasic — Levonorgestrel ─────────────────────────────────
    {
        "combo_id":           "EE20_LNG90",
        "pill_type":          "combined_monophasic",
        "estrogen":           "ethinyl estradiol",
        "estrogen_dose_mcg":  20,
        "progestin":          "levonorgestrel",
        "progestin_dose_mg":  0.09,
        "progestin_generation": 2,
        "androgenic_activity":  "moderate",
        "vte_risk_class":       "moderate",   # 2nd-gen moderate vs 3rd-gen
        "notes": "Ultra-low-dose pill. Used in Alesse, Aviane, Lutera, Orsythia, Sronyx, Lessina, Aubra",
        "brand_aliases": [
            "alesse", "aviane", "lutera", "orsythia", "sronyx", "lessina", "aubra",
            "amethyst", "dolishale", "ethinyl estradiol / levonorgestrel",
            "ethinyl estradiol / levonorgestrel 20/100",
        ],
    },
    {
        "combo_id":           "EE30_LNG150",
        "pill_type":          "combined_monophasic",
        "estrogen":           "ethinyl estradiol",
        "estrogen_dose_mcg":  30,
        "progestin":          "levonorgestrel",
        "progestin_dose_mg":  0.15,
        "progestin_generation": 2,
        "androgenic_activity":  "moderate",
        "vte_risk_class":       "moderate",
        "notes": "Standard-dose 2nd-gen. Nordette, Levora, Portia, Cryselle-28, Chateal, Jolessa (extended)",
        "brand_aliases": [
            "levora", "portia", "cryselle", "chateal", "jolessa", "seasonique",
            "camrese", "introvale", "quasense", "ethinyl estradiol / levonorgestrel",
            "ethinyl estradiol / norgestrel",  # norgestrel metabolizes to levonorgestrel
            "levonorgestrel",  # drugscom sometimes uses generic only
        ],
    },
    # ── Combined — Norethindrone (1st gen) ───────────────────────────────────
    {
        "combo_id":           "EE35_NET500_1000",
        "pill_type":          "combined_monophasic",
        "estrogen":           "ethinyl estradiol",
        "estrogen_dose_mcg":  35,
        "progestin":          "norethindrone",
        "progestin_dose_mg":  0.5,           # 0.5-1.0 mg range
        "progestin_generation": 1,
        "androgenic_activity":  "moderate",
        "vte_risk_class":       "low_moderate",
        "notes": "Classic 35mcg pill. Ortho-Novum, Necon, Nortrel, Norinyl, Sprintec variants, Mononessa",
        "brand_aliases": [
            "sprintec", "mononessa", "tri-sprintec", "trinessa", "estarylla",
            "tri-previfem", "ortho tri-cyclen", "ortho tri-cyclen lo",
            "ethinyl estradiol / norethindrone",
            "ethinyl estradiol / norgestimate",  # norgestimate -> norgestrel -> norethindrone metabolite but treated separately below
        ],
    },
    {
        "combo_id":           "EE20_NET1000",
        "pill_type":          "combined_monophasic",
        "estrogen":           "ethinyl estradiol",
        "estrogen_dose_mcg":  20,
        "progestin":          "norethindrone acetate",
        "progestin_dose_mg":  1.0,
        "progestin_generation": 1,
        "androgenic_activity":  "moderate",
        "vte_risk_class":       "low_moderate",
        "notes": "Low-estrogen 20mcg with NE acetate. Loestrin, Junel, Microgestin, Gildess, Blisovi, Lo Loestrin Fe",
        "brand_aliases": [
            "lo loestrin fe", "loestrin 24 fe", "junel fe 1 / 20", "microgestin fe 1 / 20",
            "gildess fe 1 / 20", "blisovi 24 fe", "minastrin 24 fe", "generess fe",
            "junel fe 1/20", "microgestin fe 1/20",
        ],
    },
    # ── Combined — Norgestimate (3rd gen, low androgenic) ────────────────────
    {
        "combo_id":           "EE25_35_NGM",
        "pill_type":          "combined_mono_triphasic",
        "estrogen":           "ethinyl estradiol",
        "estrogen_dose_mcg":  25,            # 25/25/25 or 35 depending on brand
        "progestin":          "norgestimate",
        "progestin_dose_mg":  0.215,         # weighted avg 0.18/0.215/0.25
        "progestin_generation": 3,
        "androgenic_activity":  "low",
        "vte_risk_class":       "moderate",  # 3rd-gen slightly higher than 2nd-gen
        "notes": "Good for acne. Ortho Tri-Cyclen, Sprintec, Tri-Sprintec, TriNessa, Tri-Previfem, Mononessa, Estarylla",
        "brand_aliases": [
            "ethinyl estradiol / norgestimate", "sprintec", "tri-sprintec", "trinessa",
            "tri-previfem", "mononessa", "estarylla", "ortho tri-cyclen", "ortho tri-cyclen lo",
        ],
    },
    # ── Combined — Desogestrel (3rd gen) ─────────────────────────────────────
    {
        "combo_id":           "EE30_DSG150",
        "pill_type":          "combined_monophasic",
        "estrogen":           "ethinyl estradiol",
        "estrogen_dose_mcg":  30,
        "progestin":          "desogestrel",
        "progestin_dose_mg":  0.15,
        "progestin_generation": 3,
        "androgenic_activity":  "low",
        "vte_risk_class":       "high",      # DSG has highest VTE risk among COCs
        "notes": "Higher VTE risk than LNG/NET. Apri, Reclipsen, Kariva, Desogen, Ortho-Cept",
        "brand_aliases": [
            "desogestrel / ethinyl estradiol", "apri", "reclipsen", "kariva",
            "desogen", "ortho-cept", "cyclessa",
        ],
    },
    # ── Combined — Drospirenone (4th gen, anti-androgenic) ───────────────────
    {
        "combo_id":           "EE30_DRSP3",
        "pill_type":          "combined_monophasic",
        "estrogen":           "ethinyl estradiol",
        "estrogen_dose_mcg":  30,
        "progestin":          "drospirenone",
        "progestin_dose_mg":  3.0,
        "progestin_generation": 4,
        "androgenic_activity":  "anti_androgenic",
        "vte_risk_class":       "high",      # DRSP highest VTE risk; FAERS full of PE/DVT
        "notes": "Anti-androgenic: good for acne/PCOS but highest thrombosis signal. Yasmin, Ocella, Zarah",
        "brand_aliases": [
            "drospirenone / ethinyl estradiol", "yasmin", "ocella", "zarah",
            "syeda", "loryna", "nikki", "gianvi",
            "drospirenone / ethinyl estradiol / levomefolate calcium", "beyaz",
        ],
    },
    {
        "combo_id":           "EE20_DRSP3",
        "pill_type":          "combined_monophasic",
        "estrogen":           "ethinyl estradiol",
        "estrogen_dose_mcg":  20,
        "progestin":          "drospirenone",
        "progestin_dose_mg":  3.0,
        "progestin_generation": 4,
        "androgenic_activity":  "anti_androgenic",
        "vte_risk_class":       "high",
        "notes": "Low-dose DRSP. Yaz, Gianvi, Vestura, Loryna",
        "brand_aliases": [
            "yaz", "gianvi", "vestura", "loryna", "nikki",
        ],
    },
    # ── Progestin-only (minipill) ─────────────────────────────────────────────
    {
        "combo_id":           "NET_PO_350",
        "pill_type":          "progestin_only",
        "estrogen":           None,
        "estrogen_dose_mcg":  0,
        "progestin":          "norethindrone",
        "progestin_dose_mg":  0.35,
        "progestin_generation": 1,
        "androgenic_activity":  "moderate",
        "vte_risk_class":       "very_low",  # no estrogen = very low VTE
        "notes": "Safe for smokers, hypertension, migraineurs, breastfeeding. Less effective if not taken consistently.",
        "brand_aliases": [
            "norethindrone", "jolivette", "nora-be", "ortho micronor", "camila",
            "errin", "heather", "jencycla", "tulana",
        ],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
#  2. SIDE EFFECT KEYWORDS to scan in drugscom free-text reviews
#     Each key = output column; values = keywords that count as a hit
# ─────────────────────────────────────────────────────────────────────────────
SIDE_EFFECT_KEYWORDS = {
    # ── Mood / mental health ──────────────────────────────────────────────────
    "kw_mood_changes":   ["mood", "mood swing", "emotional", "irritable", "irritability", "anxious", "anxiety", "mental health"],
    "kw_depression":     ["depress", "depressed", "depression", "sad", "suicidal", "crying", "low mood"],
    "kw_libido_decrease":["libido", "sex drive", "no desire", "lost interest in sex", "low libido"],
    # ── Physical ─────────────────────────────────────────────────────────────
    "kw_weight_gain":    ["weight gain", "gained weight", "weight increase", "heavier"],
    "kw_weight_loss":    ["weight loss", "lost weight", "weight decrease"],
    "kw_nausea":         ["nausea", "nauseous", "sick to my stomach", "vomit", "queasy"],
    "kw_headache":       ["headache", "migraine", "head pain"],
    "kw_breast_tenderness": ["breast", "boob", "chest tender", "breast pain", "breast sore"],
    "kw_acne_worsened":  ["acne worse", "broke out", "breakout", "pimple", "skin worse", "cystic"],
    "kw_acne_improved":  ["acne better", "acne clear", "skin clear", "cleared my skin", "no more acne", "acne gone"],
    "kw_spotting":       ["spotting", "breakthrough bleed", "irregular bleed", "light bleed between"],
    "kw_heavy_period":   ["heavy period", "heavier period", "heavy bleeding", "heavy flow", "heavy menstrual"],
    "kw_light_period":   ["light period", "lighter period", "no period", "missed period", "period stopped"],
    "kw_cramps":         ["cramp", "cramping", "painful period", "dysmenorrhea"],
    "kw_hair_loss":      ["hair loss", "hair thinning", "alopecia", "losing hair"],
    "kw_blood_clot":     ["blood clot", "dvt", "deep vein", "thrombosis", "pulmonary embolism", "clot"],
    # ── Overall satisfaction ──────────────────────────────────────────────────
    "kw_positive_overall": ["love", "great", "wonderful", "amazing", "highly recommend", "best pill", "no side effect"],
    "kw_discontinued":  ["stopped taking", "switched", "switch", "went off", "came off", "discontinued", "quit taking"],
}

# ─────────────────────────────────────────────────────────────────────────────
#  3. FAERS serious outcome terms to count per pill group
# ─────────────────────────────────────────────────────────────────────────────
FAERS_SERIOUS_TERMS = {
    "faers_dvt":          ["deep vein thrombosis", "dvt"],
    "faers_pe":           ["pulmonary embolism"],
    "faers_stroke":       ["cerebrovascular accident", "stroke", "cerebral thrombosis",
                           "cerebral venous thrombosis", "intracranial venous sinus thrombosis"],
    "faers_mi":           ["myocardial infarction", "heart attack"],
    "faers_depression":   ["depression", "depressed mood", "major depression"],
    "faers_headache":     ["headache"],
    "faers_nausea":       ["nausea"],
    "faers_weight_gain":  ["weight increased"],
    "faers_amenorrhea":   ["amenorrhoea", "menstruation irregular", "metrorrhagia"],
    "faers_anxiety":      ["anxiety"],
    "faers_alopecia":     ["alopecia"],
    "faers_libido":       ["libido decreased", "decreased libido", "loss of libido"],
}

# Maps FAERS suspect_drugs values (uppercase) → combo_id
FAERS_DRUG_MAP = {
    # ── Drospirenone combined pills (4th gen) ─────────────────────────────────
    "DROSPIRENONE AND ETHINYL ESTRADIOL":        ["EE30_DRSP3", "EE20_DRSP3"],
    "DROSPIRENONE/ETHINYL ESTRADIOL/LEVOMEFOLATE CALCIUM AND LEVOMEFOLATE CALCIUM": ["EE30_DRSP3"],
    "GIANVI":                                    ["EE20_DRSP3"],
    "DROSPIRENONE AND ETHINYL ESTRADIOL TABLETS": ["EE30_DRSP3", "EE20_DRSP3"],
    "DROSPIRENONE W/ETHINYLESTRADIOL":           ["EE30_DRSP3", "EE20_DRSP3"],
    "DROSPIRENONE\\ETHINYL ESTRADIOL":           ["EE30_DRSP3", "EE20_DRSP3"],
    "DROSPIRENONE + ETHINYLESTRADIOL":           ["EE30_DRSP3", "EE20_DRSP3"],
    # ── Levonorgestrel combined pills (2nd gen) ───────────────────────────────
    "LEVONORGESTREL":                            ["EE20_LNG90", "EE30_LNG150"],
    "LEVONORGESTREL AND ETHINYL ESTRADIOL":      ["EE20_LNG90", "EE30_LNG150"],
    # ── Norgestrel (prodrug of levonorgestrel) ────────────────────────────────
    "NORGESTREL":                                ["EE30_LNG150"],
    "NORGESTREL AND ETHINYL ESTRADIOL":          ["EE30_LNG150"],
    # ── Norgestimate combined pills (3rd gen) ─────────────────────────────────
    "NORGESTIMATE AND ETHINYL ESTRADIOL":        ["EE25_35_NGM"],
    "NORGESTIMATE/ETHINYL ESTRADIOL":            ["EE25_35_NGM"],
    "ETHINYL ESTRADIOL\\NORGESTIMATE":           ["EE25_35_NGM"],
    # ── Desogestrel combined pills (3rd gen) ─────────────────────────────────
    "DESOGESTREL AND ETHINYL ESTRADIOL":                           ["EE30_DSG150"],
    "DESOGESTREL AND ETHINYL ESTRADIOL AND ETHINYL ESTRADIOL":     ["EE30_DSG150"],
    "DESOGESTREL/ETHINYL ESTRADIOL AND ETHINYL ESTRADIOL":         ["EE30_DSG150"],
    "DESOGESTREL/ ETHINYL ESTRADIOL":                              ["EE30_DSG150"],
    "DESOGESTREL/ETHINYL ESTRADIOL":                               ["EE30_DSG150"],
    "DESOGESTREL (+) ETHINYL ESTRADIOL":                           ["EE30_DSG150"],
    # ── Norethindrone combined pills — EE35 range (1st gen) ───────────────────
    "NORETHINDRONE AND ETHINYL ESTRADIOL":                               ["EE35_NET500_1000"],
    "NORETHINDRONE AND ETHINYL ESTRADIOL AND FERROUS FUMARATE":          ["EE35_NET500_1000"],
    "NORETHINDRONE AND ETHINYL ESTRADIOL AND FERROUS FUMARATE TABLET":   ["EE35_NET500_1000"],
    "ETHINYL ESTRADIOL\\NORETHINDRONE":                                  ["EE35_NET500_1000"],
    # ── Norethindrone acetate combined pills — EE20 (1st gen) ────────────────
    "NORETHINDRONE ACETATE AND ETHINYL ESTRADIOL, ETHINYL ESTRADIOL AND FERROUS FUMARATE": ["EE20_NET1000"],
    "NORETHINDRONE ACETATE AND ETHINYL ESTRADIOL AND FERROUS FUMARATE":       ["EE20_NET1000"],
    "NORETHINDRONE ACETATE AND ETHINYL ESTRADIOL":                            ["EE20_NET1000"],
    "NORETHINDRONE ACETATE AND ETHINYL ESTRADIOL, AND FERROUS FUMARATE":      ["EE20_NET1000"],
    "NORETHINDRONE ACETATE AND ETHINYL ESTRADIOL TABLETS AND FERROUS FUMARATE TABLETS": ["EE20_NET1000"],
    "NORETHINDRONE ACETATE/ETHINYL ESTRADIOL AND FERROUS FUMARATE":           ["EE20_NET1000"],
    "NORETHINDRONE ACETATE/ETHINYL ESTRADIOL":                                ["EE20_NET1000"],
    # ── Norethindrone standalone — progestin-only mini-pill ───────────────────
    # Note: 428 FAERS reports; mostly POP but some noise from endometriosis use
    "NORETHINDRONE":                             ["NET_PO_350"],
    # ── Estradiol + norethindrone — HRT, not OCP → skip ──────────────────────
    "ESTRADIOL/NORETHINDRONE ACETATE":                           [],
    "ESTRADIOL AND NORETHINDRONE ACETATE":                       [],
    "ESTRADIOL NORETHINDRONE":                                   [],
    "ESTRADIOL AND NORETHINDRONE ACETATE TRANSDERMAL SYSTEM":    [],
    "DROSPIRENONE AND ESTRADIOL":                                [],   # Angeliq — HRT
    "ESTRADIOL AND LEVONORGESTREL":                              [],   # HRT patch
    # ── Non-pill contraceptives / abortifacients → skip ──────────────────────
    "ETONOGESTREL AND ETHINYL ESTRADIOL":        [],  # ring (NuvaRing), skip
    "MISOPROSTOL":                               [],
    "MIFEPRISTONE":                              [],
}


def normalize_drug_name(name: str) -> str:
    return name.strip().lower().replace("  ", " ")


def count_keywords(text: str, keywords: list[str]) -> int:
    text = text.lower()
    return sum(1 for kw in keywords if kw in text)


def get_rating_bucket(rating_str: str) -> str:
    try:
        r = float(rating_str)
        if r <= 3:
            return "low"
        if r <= 6:
            return "mid"
        return "high"
    except ValueError:
        return "unknown"


def load_drugscom_reviews(drugs_dir: Path) -> list[dict]:
    rows = []
    for fname in ["drugsComTrain_raw.csv", "drugsComTest_raw.csv"]:
        with open(drugs_dir / fname, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("condition", "").strip().lower() == "birth control":
                    row["review_clean"] = html.unescape(row.get("review", "")).lower()
                    row["drug_norm"] = normalize_drug_name(row.get("drugName", ""))
                    rows.append(row)
    return rows


def load_faers_reactions(drugs_dir: Path) -> list[dict]:
    rows = []
    with open(drugs_dir / "2 and 1_2 Hackers_fda_faers_events_faers_reactions_exploded.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_fda_labels(drugs_dir: Path) -> dict[str, dict]:
    """Returns a dict: substance_name -> merged label data."""
    EXCLUDE_SUBS = {"MISOPROSTOL", "MIFEPRISTONE", "DICLOFENAC SODIUM",
                    "RELUGOLIX", "ESTRADIOL", "ESTRADIOL HEMIHYDRATE",
                    "SEGESTERONE ACETATE", "DICLOFENAC"}
    by_substance: dict[str, dict] = {}
    with open(drugs_dir / "2 and 1_2 Hackers_fda_drug_labels_drug_labels_flat.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subs = row.get("substance_name", "")
            if any(ex in subs.upper() for ex in EXCLUDE_SUBS):
                continue
            # For each substance in the row, accumulate text
            for sub in subs.split("|"):
                sub = sub.strip().upper()
                if not sub:
                    continue
                if sub not in by_substance:
                    by_substance[sub] = {
                        "adverse_reactions": [],
                        "contraindications": [],
                        "warnings": [],
                        "boxed_warning": [],
                    }
                for field in ["adverse_reactions", "contraindications", "warnings", "boxed_warning"]:
                    text = row.get(field, "").strip()
                    if text:
                        by_substance[sub][field].append(text)
    # Collapse to single strings
    return {
        sub: {k: " // ".join(set(v))[:2000] for k, v in data.items()}
        for sub, data in by_substance.items()
    }


def build_reference_db(drugs_dir: Path, out_dir: Path):
    print("Loading drugscom reviews...")
    reviews = load_drugscom_reviews(drugs_dir)
    print(f"  {len(reviews)} birth control reviews loaded")

    print("Loading FAERS reactions...")
    faers = load_faers_reactions(drugs_dir)
    print(f"  {len(faers)} FAERS reaction rows loaded")

    print("Loading FDA labels...")
    fda_labels = load_fda_labels(drugs_dir)
    print(f"  {len(fda_labels)} substance label entries")

    # ── Index reviews by alias ───────────────────────────────────────────────
    alias_to_combo: dict[str, str] = {}
    for pill in PILL_MASTER:
        for alias in pill["brand_aliases"]:
            alias_to_combo[normalize_drug_name(alias)] = pill["combo_id"]

    # ── Index FAERS by combo_id ───────────────────────────────────────────────
    faers_by_combo: dict[str, list[str]] = collections.defaultdict(list)
    for row in faers:
        drugs_in_row = [d.strip().upper() for d in row.get("suspect_drugs", "").split("|")]
        term = row.get("reaction_term", "").strip().lower()
        if not term:
            continue
        for drug in drugs_in_row:
            for combo_ids in [FAERS_DRUG_MAP.get(drug, [])]:
                for cid in combo_ids:
                    faers_by_combo[cid].append(term)

    # ── Build one output row per canonical formulation ───────────────────────
    output_rows = []

    for pill in PILL_MASTER:
        cid = pill["combo_id"]
        print(f"  Processing {cid}...")

        # Match reviews
        pill_reviews = [
            r for r in reviews
            if alias_to_combo.get(r["drug_norm"]) == cid
        ]

        n_reviews = len(pill_reviews)
        ratings = [float(r["rating"]) for r in pill_reviews if r.get("rating")]
        avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None

        # Keyword side-effect frequencies (as % of reviews that mention it)
        kw_freqs = {}
        for col, keywords in SIDE_EFFECT_KEYWORDS.items():
            if n_reviews > 0:
                hits = sum(
                    1 for r in pill_reviews
                    if count_keywords(r["review_clean"], keywords) > 0
                )
                kw_freqs[col] = round(hits / n_reviews, 4)
            else:
                kw_freqs[col] = None

        # FAERS signal counts (absolute counts — normalize later per model)
        faers_terms = faers_by_combo.get(cid, [])
        faers_total = len(faers_terms)
        faers_counts = {}
        for col, terms in FAERS_SERIOUS_TERMS.items():
            count = sum(1 for t in faers_terms if any(term in t for term in terms))
            faers_counts[col] = count
        faers_counts["faers_total_reports"] = faers_total

        # FDA label text — look up by progestin + estrogen names
        label_data = {}
        progestin_key = pill["progestin"].upper().replace(" ", " ") if pill["progestin"] else ""
        for field in ["adverse_reactions", "contraindications", "warnings", "boxed_warning"]:
            texts = []
            if progestin_key and progestin_key in fda_labels:
                t = fda_labels[progestin_key].get(field, "")
                if t:
                    texts.append(t)
            if "ETHINYL ESTRADIOL" in fda_labels and pill.get("estrogen"):
                t = fda_labels["ETHINYL ESTRADIOL"].get(field, "")
                if t:
                    texts.append(t)
            label_data[f"label_{field}"] = " // ".join(texts)[:1500]

        row = {
            # Identity
            "combo_id":              cid,
            "pill_type":             pill["pill_type"],
            "estrogen":              pill["estrogen"] or "",
            "estrogen_dose_mcg":     pill["estrogen_dose_mcg"],
            "progestin":             pill["progestin"] or "",
            "progestin_dose_mg":     pill["progestin_dose_mg"],
            "progestin_generation":  pill["progestin_generation"],
            "androgenic_activity":   pill["androgenic_activity"],
            "vte_risk_class":        pill["vte_risk_class"],
            "notes":                 pill["notes"],
            "known_brand_examples":  ", ".join(pill["brand_aliases"][:5]),
            # drugscom stats
            "drugscom_n_reviews":    n_reviews,
            "drugscom_avg_rating":   avg_rating,
            **kw_freqs,
            # FAERS stats
            **faers_counts,
            # FDA label text
            **label_data,
        }
        output_rows.append(row)

    # ── Write output ─────────────────────────────────────────────────────────
    out_path = out_dir / "pill_reference_db.csv"
    if output_rows:
        fieldnames = list(output_rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(output_rows)

    print(f"\n✓ pill_reference_db.csv written to: {out_path}")
    print(f"  {len(output_rows)} canonical pill formulations")
    print()

    # ── Summary stats ────────────────────────────────────────────────────────
    print("Coverage summary:")
    for r in output_rows:
        print(f"  {r['combo_id']:<20}  {r['drugscom_n_reviews']:>5} drugscom reviews  "
              f"{r['faers_total_reports']:>5} FAERS reports  "
              f"avg_rating={r['drugscom_avg_rating']}")

    return output_rows


if __name__ == "__main__":
    build_reference_db(DRUGS_DIR, OUT_DIR)
