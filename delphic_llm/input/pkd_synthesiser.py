"""
DELPHIC-LLM: PKD Synthesiser — Enhanced version
Converts NASA93 CSV rows to rich natural language Project Knowledge Documents.

Key improvements over v1:
- Rich contextual descriptions (mode, centre, year, COCOMO interpretation)
- Explicit KLOC-to-effort calibration ranges from NASA93 statistics
- COCOMO base estimate included as anchor (helps GPT-4o calibrate scale)
- Full cost driver narrative (not just labels)
"""
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from delphic_llm.models import ProjectKnowledgeDocument
except ImportError:
    class ProjectKnowledgeDocument:
        def __init__(self, **kw): self.__dict__.update(kw)
        def to_context_string(self):
            parts = []
            for f in ['pkd1_context','pkd2_scope','pkd3_resources',
                      'pkd6_known_unknowns']:
                v = getattr(self, f, '')
                if v: parts.append(v)
            return '\n\n'.join(parts)

# ── Label mappings ─────────────────────────────────────────────────────────────
LABEL_MAP = {
    'vl': (1, 'very low'),   'l': (2, 'low'),
    'n':  (3, 'nominal'),    'h': (4, 'high'),
    'vh': (5, 'very high'),  'xh': (6, 'extra high'),
}

COST_DRIVER_COLS = [
    'rely','data','cplx','time','stor','virt','turn',
    'acap','aexp','pcap','vexp','lexp','modp','tool','sced'
]

CAT2_DESCRIPTIONS = {
    'avionicsmonitoring': 'Avionics monitoring',
    'Avionics':           'Avionics',
    'missionplanning':    'Mission planning',
    'monitor_control':    'Monitor and control',
    'operatingsystem':    'Operating system',
    'simulation':         'Simulation',
    'science':            'Scientific/mathematical',
    'batchdataprocessing':'Batch data processing',
    'realdataprocessing': 'Real-time data processing',
    'datacapture':        'Data capture',
    'communications':     'Communications',
    'launchprocessing':   'Launch processing',
    'utility':            'Utility software',
    'application_ground': 'Ground application',
}

CENTER_NAMES = {
    '1': 'Goddard Space Flight Center',
    '2': 'Johnson Space Center',
    '3': 'Marshall Space Flight Center',
    '5': 'Jet Propulsion Laboratory',
    '6': 'Kennedy Space Center',
}

MODE_DESCRIPTIONS = {
    'organic':      ('Organic', 'small experienced team, familiar environment, flexible requirements'),
    'semidetached': ('Semidetached', 'medium team with mixed experience, some rigid requirements'),
    'embedded':     ('Embedded', 'tight hardware/software/operational constraints, most requirements fixed'),
}

# COCOMO intermediate parameters by mode (a, b)
COCOMO_PARAMS = {
    'organic': (2.4, 1.05), 'semidetached': (3.0, 1.12), 'embedded': (3.6, 1.20)
}

# Empirical effort ranges from actual NASA93 data (person-hours)
# Used to give GPT-4o a calibration anchor for scale
KLOC_EFFORT_RANGES = {
    '<5':    (1277,   1459,   5776),
    '5-15':  (1824,   7296,  98496),
    '15-30': (7296,  17602,  72960),
    '30-75': (9120,  45600, 292524),
    '75-200':(14744, 66576, 635086),
    '>200':  (29184,207936,1248072),
}

HOURS_PER_PERSON_MONTH = 152


def lvl(val: str) -> str:
    if isinstance(val, str):
        c = val.strip().lower()
        if c in LABEL_MAP:
            return LABEL_MAP[c][1]
    return 'nominal'


def lvl_int(val: str) -> int:
    if isinstance(val, str):
        c = val.strip().lower()
        if c in LABEL_MAP:
            return LABEL_MAP[c][0]
    return 3


def cocomo_base_estimate(kloc: float, mode: str) -> tuple:
    """Return (person_months, person_hours) COCOMO base estimate (no EAF)."""
    a, b = COCOMO_PARAMS.get(str(mode).strip().lower(), (3.0, 1.12))
    pm = a * (kloc ** b)
    return round(pm, 1), round(pm * HOURS_PER_PERSON_MONTH, 0)


def kloc_range_label(kloc: float) -> str:
    if kloc < 5:    return '<5'
    if kloc < 15:   return '5-15'
    if kloc < 30:   return '15-30'
    if kloc < 75:   return '30-75'
    if kloc < 200:  return '75-200'
    return '>200'


def describe_eaf_profile(row) -> str:
    """Summarise which cost drivers are above/below nominal."""
    above = []
    below = []
    for col in COST_DRIVER_COLS:
        val = lvl_int(str(row.get(col, 'n')))
        name = col.upper()
        if val >= 5:   above.append(f"{name}={lvl(str(row.get(col,'n')))}")
        elif val <= 2: below.append(f"{name}={lvl(str(row.get(col,'n')))}")
    parts = []
    if above: parts.append(f"HIGH-EFFORT DRIVERS: {', '.join(above)}")
    if below: parts.append(f"LOW-EFFORT DRIVERS:  {', '.join(below)}")
    if not parts: parts.append("All cost drivers at nominal levels")
    return '\n'.join(parts)


def synthesise_pkd_nasa93(row: pd.Series) -> 'ProjectKnowledgeDocument':
    """
    Convert one NASA93 row to a rich ProjectKnowledgeDocument.
    Includes COCOMO base estimate and empirical effort ranges
    to help GPT-4o calibrate its predictions correctly.
    """
    # Basic attributes
    cat2      = str(row.get('cat2', '')).strip()
    app_desc  = CAT2_DESCRIPTIONS.get(cat2, cat2.replace('_', ' ').title())
    center_id = str(row.get('center', '')).strip()
    center    = CENTER_NAMES.get(center_id, f'NASA Centre {center_id}')
    year      = str(int(row.get('year', 1980)))
    mode_raw  = str(row.get('mode', 'semidetached')).strip().lower()
    mode_short, mode_desc = MODE_DESCRIPTIONS.get(mode_raw, ('Semidetached', mode_raw))
    
    try:
        kloc = float(row.get('equivphyskloc', 10))
        if kloc >= 100:
            size_str = f"{kloc:.0f} KLOC ({int(kloc*1000):,} equivalent LOC)"
        elif kloc >= 1:
            size_str = f"{kloc:.1f} KLOC ({int(kloc*1000):,} equivalent LOC)"
        else:
            size_str = f"{kloc:.2f} KLOC ({int(kloc*1000)} equivalent LOC)"
    except (TypeError, ValueError):
        kloc = 10.0
        size_str = "size not specified"

    # COCOMO base estimate — critical calibration anchor
    base_pm, base_hrs = cocomo_base_estimate(kloc, mode_raw)
    
    # Empirical range for this KLOC band
    range_key  = kloc_range_label(kloc)
    rng_min, rng_med, rng_max = KLOC_EFFORT_RANGES[range_key]
    
    # EAF profile — which drivers push effort up or down
    eaf_summary = describe_eaf_profile(row)

    # Schedule note
    sced_int = lvl_int(str(row.get('sced', 'n')))
    sched_note = ('compressed schedule required — adds overhead' if sced_int <= 2
                  else 'relaxed schedule — allows phased delivery' if sced_int >= 5
                  else 'nominal schedule constraint')

    # ── PKD₁: Project context ─────────────────────────────────────────────────
    pkd1 = f"""NASA software development project — {app_desc} application.

Organisation: {center}, {year}.
Development mode: {mode_short} mode ({mode_desc}).
Domain: {app_desc} software for NASA mission support.

SCALE REFERENCE — IMPORTANT FOR ESTIMATION:
This project is {size_str}.
For NASA {mode_short.lower()} mode projects of this size ({range_key} KLOC),
historical effort in this dataset ranges from {rng_min:,}h to {rng_max:,}h,
with a typical (median) value of approximately {rng_med:,} person-hours.

The COCOMO intermediate model base estimate for this size and mode
(before applying cost driver multipliers) is approximately {base_pm} person-months
= {base_hrs:,.0f} person-hours. Actual effort will be higher or lower
depending on the cost driver profile below."""

    # ── PKD₂: Technical scope ─────────────────────────────────────────────────
    pkd2 = f"""Technical scope and constraints (COCOMO intermediate cost drivers):

Project size: {size_str}
Development mode: {mode_short} ({mode_desc})

COST DRIVER PROFILE:
Required software reliability:        {lvl(row.get('rely','n'))}
Product complexity:                   {lvl(row.get('cplx','n'))}
Database size relative to program:    {lvl(row.get('data','n'))}
Execution time constraint:            {lvl(row.get('time','n'))}
Main storage constraint:              {lvl(row.get('stor','n'))}
Virtual machine (platform) volatility:{lvl(row.get('virt','n'))}
Computer turnaround time:             {lvl(row.get('turn','n'))}

EFFORT IMPACT SUMMARY:
{eaf_summary}
Schedule constraint: {lvl(row.get('sced','n'))} ({sched_note})"""

    # ── PKD₃: Team and resources ──────────────────────────────────────────────
    pkd3 = f"""Team capability and experience profile (COCOMO cost drivers):

Analyst capability:                   {lvl(row.get('acap','n'))}
Analyst application domain experience:{lvl(row.get('aexp','n'))}
Programmer capability:                {lvl(row.get('pcap','n'))}
Virtual machine experience:           {lvl(row.get('vexp','n'))}
Programming language experience:      {lvl(row.get('lexp','n'))}

Development practices:
Use of modern programming practices:  {lvl(row.get('modp','n'))}
Use of software tools:                {lvl(row.get('tool','n'))}

Note: Capability ratings above nominal reduce effort; below nominal increase it.
Nominal = average performance for NASA software development in this era."""

    # ── PKD₆: Known unknowns ──────────────────────────────────────────────────
    pkd6 = f"""Known unknowns and estimation limitations:
- This is a historical archived project; detailed requirements not available
- Team headcount and individual assignments not recorded
- Budget ceiling not specified
- The COCOMO cost driver ratings summarise the project but do not capture
  all nuances of the original project context
- PKD4 (visual artefacts) and PKD5 (historical analogues) not available
  for this dataset — estimate without analogical reference data

ESTIMATION GUIDANCE:
Your estimate should be anchored to the scale reference in PKD1.
A reasonable estimate for a {kloc:.1f} KLOC {mode_short.lower()} mode project
with this cost driver profile is in the range of {rng_min:,}–{rng_max:,} person-hours.
The COCOMO base of {base_hrs:,.0f}h should be adjusted up or down based on
the cost driver profile. Do not estimate outside the range
{max(500, rng_min//3):,}–{rng_max*3:,} hours without explicit justification."""

    return ProjectKnowledgeDocument(
        pkd1_context=pkd1,
        pkd2_scope=pkd2,
        pkd3_resources=pkd3,
        pkd4_visuals="",
        pkd5_analogues="",
        pkd6_known_unknowns=pkd6
    )


def load_nasa93(filepath: str = None) -> pd.DataFrame:
    if filepath is None:
        for candidate in ['data/nasa93.csv', 'nasa93.csv']:
            if Path(candidate).exists():
                filepath = candidate
                break
        if filepath is None:
            raise FileNotFoundError("NASA93 not found. Place at data/nasa93.csv")

    df = pd.read_csv(filepath)
    required = ['act_effort', 'equivphyskloc'] + COST_DRIVER_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df['effort_hours'] = pd.to_numeric(df['act_effort'], errors='coerce') * HOURS_PER_PERSON_MONTH
    df['loc_kloc']     = pd.to_numeric(df['equivphyskloc'], errors='coerce')
    df = df.dropna(subset=['effort_hours'])
    df = df[df['effort_hours'] > 0].reset_index(drop=True)
    print(f"  Loaded {len(df)} NASA93 records | "
          f"effort range: {df.effort_hours.min():.0f}–{df.effort_hours.max():.0f} hours | "
          f"{df['cat2'].nunique()} project types")
    return df


def get_stratified_sample(df: pd.DataFrame, n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    type_counts = df['cat2'].value_counts()

    if n <= len(type_counts):
        selected = type_counts.head(n).index.tolist()
        parts = []
        for cat in selected:
            group = df[df['cat2'] == cat]
            idx = rng.choice(len(group), size=1, replace=False)
            parts.append(group.iloc[idx])
        return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    allocations = {}
    for cat, count in type_counts.items():
        allocations[cat] = max(1, round(n * count / len(df)))
        allocations[cat] = min(allocations[cat], count)

    cats = type_counts.index.tolist()
    idx = 0
    while sum(allocations.values()) > n:
        cat = cats[idx % len(cats)]
        if allocations[cat] > 1: allocations[cat] -= 1
        idx += 1
        if idx > len(cats) * 200: break

    idx = 0
    while sum(allocations.values()) < n:
        cat = cats[idx % len(cats)]
        if allocations[cat] < type_counts[cat]: allocations[cat] += 1
        idx += 1
        if idx > len(cats) * 200: break

    parts = []
    for cat, count in allocations.items():
        group = df[df['cat2'] == cat]
        take = min(count, len(group))
        i = rng.choice(len(group), size=take, replace=False)
        parts.append(group.iloc[i])

    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True).iloc[:n]


def prepare_nasa93_dataset(filepath: str = None, n_sample: int = 50,
                            seed: int = 42) -> tuple:
    print("Loading NASA93...")
    df = load_nasa93(filepath)
    print(f"Stratified sampling n={n_sample} (seed={seed})...")
    sample = get_stratified_sample(df, n=n_sample, seed=seed)
    breakdown = sample['cat2'].value_counts().to_dict()
    print(f"  {len(breakdown)} types: " +
          ", ".join(f"{k}({v})" for k, v in sorted(breakdown.items())))
    print("Synthesising PKDs...")
    pkds = [synthesise_pkd_nasa93(row) for _, row in sample.iterrows()]
    print(f"  {len(pkds)} PKDs ready.\n")
    return sample, pkds
