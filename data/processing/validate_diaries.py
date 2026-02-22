"""Quick clinical plausibility validation for symptom_diaries.csv."""
import csv
from collections import defaultdict

rows = list(csv.DictReader(open('data/output/symptom_diaries.csv')))
print(f"Total rows: {len(rows):,}\n")

# ── 1. Month-1 symptom rates by pill ─────────────────────────────────────
pill_mo1 = defaultdict(list)
for r in rows:
    if r['month'] == '1' and r['still_taking'] == '1':
        pill_mo1[r['combo_id']].append(r)

syms = ['sym_nausea','sym_mood_worsened','sym_spotting','sym_acne_improved','sym_acne_worsened']
print("── Month-1 symptom rates by pill ──────────────────────────────────────")
hdr = f"{'combo_id':<22}" + "".join(f"{s.replace('sym_',''):>15}" for s in syms)
print(hdr)
for cid in sorted(pill_mo1):
    rs = pill_mo1[cid]
    n  = len(rs)
    vals = "".join(f"{100*sum(int(r[s]) for r in rs)/n:>14.1f}%" for s in syms)
    print(f"{cid:<22}{vals}  n={n}")

# ── 2. Serious events by pill ──────────────────────────────────────────────
print("\n── Serious events by pill (all months, active users) ──────────────────")
pill_pm = defaultdict(int)
pill_ev = defaultdict(lambda: defaultdict(int))
for r in rows:
    if r['still_taking'] == '1':
        cid = r['combo_id']
        pill_pm[cid] += 1
        for ev in ('evt_dvt','evt_pe','evt_stroke'):
            pill_ev[cid][ev] += int(r[ev])

for cid in sorted(pill_pm):
    pm = pill_pm[cid]
    d, p, s = pill_ev[cid]['evt_dvt'], pill_ev[cid]['evt_pe'], pill_ev[cid]['evt_stroke']
    ann_vte = 12000 * (d+p) / pm  # /10,000/yr
    print(f"  {cid:<22}  {pm:>7} pm  DVT={d}  PE={p}  stroke={s}  "
          f"approx_VTE_rate={ann_vte:.2f}/10k/yr")

# ── 3. Satisfaction by WHO MEC category (month 3) ─────────────────────────
print("\n── Mean satisfaction by WHO MEC category (month 3) ─────────────────────")
by_mec = defaultdict(list)
for r in rows:
    if r['month'] == '3' and r['still_taking'] == '1' and r['satisfaction_score']:
        by_mec[int(r['who_mec_category'])].append(float(r['satisfaction_score']))
for cat in sorted(by_mec):
    vals = by_mec[cat]
    print(f"  WHO MEC Cat {cat}: mean_satisfaction={sum(vals)/len(vals):.2f}  n={len(vals):,}")

# ── 4. DRSP vs LNG: acne improvement (month 6) ────────────────────────────
print("\n── Acne improvement month 6: anti-androgenic vs moderate ───────────────")
for cid in ('EE20_DRSP3','EE30_DRSP3','EE25_35_NGM','EE30_LNG150','NET_PO_350'):
    rs = [r for r in rows if r['combo_id']==cid
          and r['month']=='6' and r['still_taking']=='1']
    n = len(rs)
    ai = sum(int(r['sym_acne_improved']) for r in rs) / max(n,1)
    aw = sum(int(r['sym_acne_worsened']) for r in rs) / max(n,1)
    print(f"  {cid:<22}  acne_improved={100*ai:.1f}%  acne_worsened={100*aw:.1f}%  n={n}")

# ── 5. WHO4 pill vs appropriate pill: cramps_relieved (endometriosis signal) ─
print("\n── Cramps relieved (month 6, combined appropriate pills) ───────────────")
for cid in ('EE30_DRSP3','EE30_LNG150','NET_PO_350'):
    rs = [r for r in rows if r['combo_id']==cid
          and r['month']=='6' and r['still_taking']=='1']
    n  = len(rs)
    cr = sum(int(r['sym_cramps_relieved']) for r in rs) / max(n,1)
    print(f"  {cid:<22}  cramps_relieved={100*cr:.1f}%  n={n}")

# ── 6. Discontinuation by appropriateness ─────────────────────────────────
print("\n── 12-month discontinuation by prescription_appropriate ─────────────────")
for appr in ('1','0'):
    pairs = set((r['patient_id'], r['combo_id']) for r in rows
                if r['prescription_appropriate'] == appr)
    disc  = set((r['patient_id'], r['combo_id']) for r in rows
                if r['prescription_appropriate'] == appr and r['discontinued_reason'])
    label = "Appropriate (WHO 1-2)" if appr=='1' else "Inappropriate (WHO 3-4)"
    pct = 100*len(disc)/max(len(pairs),1)
    print(f"  {label:<28}: {len(disc):>6,} / {len(pairs):>6,} pairs = {pct:.1f}% discontinued")

# ── 7. Temporal nausea curve (combined pills) ─────────────────────────────
print("\n── Nausea rate by month (EE30_DRSP3, active users) ─────────────────────")
for month in range(1, 13):
    rs = [r for r in rows if r['combo_id']=='EE30_DRSP3'
          and r['month']==str(month) and r['still_taking']=='1']
    n  = len(rs)
    rate = sum(int(r['sym_nausea']) for r in rs) / max(n,1)
    bar  = '█' * int(rate * 30)
    print(f"  Month {month:>2}: {100*rate:5.1f}%  {bar}  (n={n})")
