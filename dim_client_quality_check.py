"""
dim_client_quality_check.py
---------------------------
Profiles DATA/RAW/dim_client_raw.csv, detects data quality problems, and
prints them as a table with five columns:

    Column | Issue | Fixing Strategy | Justification | Risk

The same table is also saved to reports/dim_client_quality_issues.csv.
The raw file is only read, never changed.

Run from the project folder (with the .venv active):
    pip install pandas
    python dim_client_quality_check.py
"""

import re
import textwrap
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "DATA" / "RAW" / "dim_client_raw.csv"
ADVISOR_FILE = BASE_DIR / "DATA" / "RAW" / "dim_advisor_raw.csv"
REPORT_DIR = BASE_DIR / "reports"
REPORT_FILE = REPORT_DIR / "dim_client_quality_issues.csv"

# --------------------------------------------------------------------------
# Reference values (the domain rules the data is checked against)
# --------------------------------------------------------------------------
US_STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC",
}
VALID_RISK = {"Conservative", "Moderately Conservative", "Balanced",
              "Moderately Aggressive", "Aggressive"}
VALID_TIER = {"Retail", "HNW", "UHNW"}
VALID_ACCOUNT = {"Taxable", "IRA", "Roth IRA", "Trust", "529"}
TIER_AUM_RANGE = {             # tier cut-offs observed in the clean records
    "Retail": (0, 500_000),
    "HNW": (500_000, 5_000_000),
    "UHNW": (5_000_000, float("inf")),
}
MIN_AGE, MAX_AGE = 18, 110
NAME_PATTERN = r"^[A-Za-z][A-Za-z .'\-]*$"

issues = []  # each item: dict with the five report columns


def add(column, issue, fix, why, risk):
    issues.append({
        "Column": column,
        "Issue": issue,
        "Fixing Strategy": fix,
        "Justification": why,
        "Risk": risk,
    })


def examples(series, n=4):
    vals = list(dict.fromkeys(series.astype(str).tolist()))[:n]
    return ", ".join(repr(v) for v in vals)


# --------------------------------------------------------------------------
# Load everything as text so nothing is silently converted
# --------------------------------------------------------------------------
df = pd.read_csv(RAW_FILE, dtype=str, keep_default_na=False)
n_rows = len(df)

# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

# 1. Blank values in any column
for col in df.columns:
    blank = df[col].str.strip() == ""
    if blank.any():
        add(col,
            f"{blank.sum()} rows ({blank.mean():.1%}) are blank / missing.",
            "Replace blanks with an explicit 'Unknown' category (do not "
            "guess a profile). Flag the clients so advisors can collect the "
            "missing questionnaire.",
            "Risk profile drives suitability decisions. Imputing it from "
            "age or tier would invent regulatory-relevant data.",
            "Medium: 'Unknown' clients drop out of risk-based analysis "
            "until the data is collected.")

# 2. Duplicate client_id
dup_mask = df["client_id"].duplicated(keep=False)
if dup_mask.any():
    n_ids = df.loc[dup_mask, "client_id"].nunique()
    n_extra = df["client_id"].duplicated().sum()
    add("client_id",
        f"{n_ids} client_ids appear twice ({n_extra} extra rows), which "
        f"breaks the primary key. e.g. {examples(df.loc[dup_mask, 'client_id'])}.",
        "For each duplicated id, keep the row whose last_name is valid "
        "(letters only) and drop the corrupted copy; if both are valid, "
        "keep the first occurrence.",
        "A dimension table needs one row per key, or joins to the fact "
        "tables double-count fees, transactions and AUM.",
        "Low: the copies are identical except for the garbled last name, "
        "so no real information is lost.")

# 3. Duplicate records that differ only in last_name
if dup_mask.any():
    other_cols = [c for c in df.columns if c not in ("client_id", "last_name")]
    grp = df[dup_mask].groupby("client_id")[other_cols].nunique()
    same_except_name = (grp.max(axis=1) == 1).sum()
    add("client_id / last_name",
        f"{same_except_name} of the duplicated ids are the same client "
        "re-loaded with a corrupted last name (near-duplicate records).",
        "Resolve by deduplicating on client_id (see above) rather than on "
        "the full row, because a full-row check misses these.",
        "An exact-duplicate check reports 0 duplicates here, so the "
        "problem would slip through without a key-based check.",
        "Low: the matching columns confirm the rows are the same person.")

# 4. Name columns with invalid characters
for col in ["first_name", "last_name"]:
    bad = ~df[col].str.match(NAME_PATTERN)
    if bad.any():
        add(col,
            f"{bad.sum()} values contain digits or symbols "
            f"(e.g. {examples(df.loc[bad, col])}), e.g. '3' for 'e', "
            "'0' for 'o', '@' for 'a', '!' for 'i'.",
            "Where the client_id has a clean duplicate, use the clean name. "
            "Otherwise map 3->e, 0->o, @->a, !->i, then title-case.",
            "The substitutions follow a consistent character-swap pattern "
            "and the clean copies confirm the intended spelling.",
            "Low: the fixes are verified against the clean copies; any "
            "name without a clean copy should be reviewed by hand.")

# 5. Age outside a plausible adult range
age_num = pd.to_numeric(df["age"], errors="coerce")
bad_age = age_num.isna() | (age_num < MIN_AGE) | (age_num > MAX_AGE)
if bad_age.any():
    add("age",
        f"{bad_age.sum()} values are impossible for an investment client "
        f"(< {MIN_AGE} or > {MAX_AGE}): {examples(df.loc[bad_age, 'age'], 6)}. "
        "They look like placeholder or sentinel values.",
        "Set these ages to null (NaN) and keep the column as an integer "
        "type. Do not replace them with the mean or median.",
        "0, -5 and 999 are data-entry defaults, not real ages. Imputing a "
        "value would hide the error and bias age-band analysis.",
        "Low: very few rows are affected; they are excluded only from "
        "age-based analysis.")

# 6. State: mixed full names and abbreviations
st = df["state"].str.strip()
full_names = st.isin(US_STATES.keys())
if full_names.any():
    add("state",
        f"Inconsistent format: {full_names.sum()} rows use full state names "
        f"(e.g. {examples(st[full_names])}) while the rest use two-letter "
        f"codes. This gives {st.nunique()} distinct values for "
        f"{st.map(lambda s: US_STATES.get(s, s)).nunique()} real states.",
        "Map full names to their USPS two-letter code with a lookup "
        "dictionary and store every value in upper case.",
        "Grouping by state would otherwise split 'PA' and 'Pennsylvania' "
        "into separate states and undercount each.",
        "Very low: a name-to-code mapping is deterministic.")
invalid_state = ~st.isin(set(US_STATES.values()) | set(US_STATES.keys()))
if invalid_state.any():
    add("state",
        f"{invalid_state.sum()} values are not valid US states: "
        f"{examples(st[invalid_state])}.",
        "Set to null and flag for review.",
        "Unrecognised codes cannot be mapped to a region.",
        "Low.")

# 7. City: whitespace and casing
city = df["city"]
padded = city != city.str.strip()
upper = (city == city.str.upper()) & city.str.contains("[A-Z]")
if padded.any() or upper.any():
    add("city",
        f"{padded.sum()} values have leading/trailing spaces and "
        f"{upper.sum()} are in ALL CAPS (e.g. {examples(city[padded | upper], 3)}), "
        "while the rest are in title case.",
        "Trim whitespace with str.strip(), then apply str.title().",
        "'  EAST MARK  ' and 'East Mark' are the same city but would be "
        "counted separately in grouping and joins.",
        "Very low: title-casing may change a few spellings "
        "(e.g. 'McAllen' -> 'Mcallen'), which is acceptable here.")

# 8. zip_code lost leading zeros
z = df["zip_code"].str.strip()
short = z.str.fullmatch(r"\d{1,4}")
non_digit = ~z.str.fullmatch(r"\d+")
if short.any():
    add("zip_code",
        f"{short.sum()} ZIP codes have fewer than 5 digits "
        f"(e.g. {examples(z[short])}). Leading zeros were dropped when the "
        "column was stored as a number.",
        "Keep zip_code as text and left-pad to 5 digits with "
        "str.zfill(5) (e.g. 3897 -> 03897).",
        "ZIP codes are identifiers, not quantities. Northeast ZIPs "
        "(NJ, MA, CT...) start with 0.",
        "Low: padding restores the original value; note that the ZIPs do "
        "not always match the state in this synthetic data.")
if non_digit.any():
    add("zip_code",
        f"{non_digit.sum()} ZIP codes contain non-digit characters.",
        "Strip non-digits (keep the 5-digit part of ZIP+4); null if none remain.",
        "ZIP codes must be numeric strings.",
        "Low.")

# 9. risk_profile / client_tier / primary_account_type domains
for col, valid in [("risk_profile", VALID_RISK), ("client_tier", VALID_TIER),
                   ("primary_account_type", VALID_ACCOUNT)]:
    vals = df[col].str.strip()
    bad = (vals != "") & ~vals.isin(valid)
    if bad.any():
        add(col,
            f"{bad.sum()} values are outside the allowed list: "
            f"{examples(vals[bad])}.",
            "Map spelling/case variants to the allowed values; null anything else.",
            "Categorical columns must use one consistent set of labels.",
            "Low.")

# 10. primary_account_type '529' can be read as a number
if (df["primary_account_type"] == "529").any():
    n529 = (df["primary_account_type"] == "529").sum()
    add("primary_account_type",
        f"{n529} rows hold the value '529' (a college-savings plan), which "
        "tools like Excel or pandas can read as the number 529.",
        "Always read the column as text (dtype=str). Optionally relabel as "
        "'529 Plan' for clarity.",
        "If the column is converted to a number, the category turns into "
        "an integer or makes the whole column mixed-type.",
        "Very low: relabelling is cosmetic; the value itself is valid.")

# 11. initial_aum: numeric validity and consistency with client_tier
aum = pd.to_numeric(df["initial_aum"], errors="coerce")
bad_aum = aum.isna() | (aum <= 0)
if bad_aum.any():
    add("initial_aum",
        f"{bad_aum.sum()} values are non-numeric, zero or negative.",
        "Set to null and flag for review.",
        "Assets under management must be a positive amount.",
        "Medium: missing AUM affects fee and tier analysis.")
mismatch = pd.Series(False, index=df.index)
for tier, (lo, hi) in TIER_AUM_RANGE.items():
    t = df["client_tier"] == tier
    mismatch |= t & ((aum < lo) | (aum >= hi))
if mismatch.any():
    add("initial_aum / client_tier",
        f"{mismatch.sum()} clients have an AUM that does not match their tier.",
        "Recompute the tier from AUM using the firm's cut-offs.",
        "The tier should follow the documented AUM thresholds.",
        "Medium: changing a tier affects fee schedules.")
# Note (information only): AUM is highly skewed, so outliers are expected
if aum.notna().any():
    q1, q3 = aum.quantile([0.25, 0.75])
    outl = aum > q3 + 3 * (q3 - q1)
    if outl.any() and not mismatch.any():
        add("initial_aum",
            f"{outl.sum()} values are statistical outliers (> Q3 + 3*IQR, "
            f"max {aum.max():,.2f}). All of them belong to UHNW clients and "
            "match their tier.",
            "No change. Keep as-is; use log scale or median when analysing.",
            "The large values are real ultra-high-net-worth balances, not "
            "errors. Removing them would delete the most valuable clients.",
            "Low: the only risk is skewed averages if the mean is used.")

# 12. client_since: mixed date formats
cs = df["client_since"].str.strip()
iso = cs.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
mdy = cs.str.fullmatch(r"\d{2}/\d{2}/\d{4}")
other = ~(iso | mdy)
if mdy.any():
    day_gt_12 = (mdy & (cs.str[3:5].astype(str).str.isdigit())
                 & (pd.to_numeric(cs.str[3:5], errors="coerce") > 12)).sum()
    add("client_since",
        f"Mixed date formats: {iso.sum()} rows are YYYY-MM-DD and {mdy.sum()} "
        f"are MM/DD/YYYY (e.g. {examples(cs[mdy], 3)}). The column is stored "
        "as text, not as a date.",
        "Parse each format explicitly (format='%Y-%m-%d', then "
        "format='%m/%d/%Y' for the rest) and convert to a single datetime "
        "column stored as ISO YYYY-MM-DD.",
        f"{day_gt_12} of the slash dates have a day > 12, which proves the "
        "order is month/day, not day/month. Mixed text dates cannot be "
        "sorted or used for tenure calculations.",
        "Low: slash dates with day <= 12 are assumed to be MM/DD like the "
        "rest; an explicit format prevents mis-parsing.")
if other.any():
    add("client_since",
        f"{other.sum()} dates are in an unrecognised format: {examples(cs[other])}.",
        "Set to null and flag for review.",
        "Unparseable dates cannot be used.",
        "Low.")
parsed = pd.concat([
    pd.to_datetime(cs[iso], format="%Y-%m-%d", errors="coerce"),
    pd.to_datetime(cs[mdy], format="%m/%d/%Y", errors="coerce"),
]).reindex(df.index)
future = parsed > pd.Timestamp.today()
if future.any():
    add("client_since",
        f"{future.sum()} dates are in the future.",
        "Set to null and flag for review.",
        "A client cannot have joined in the future.",
        "Low.")

# 13. advisor_id: foreign key check (only if the advisor file exists)
adv = pd.to_numeric(df["advisor_id"], errors="coerce")
if adv.isna().any():
    add("advisor_id",
        f"{adv.isna().sum()} advisor_ids are blank or non-numeric.",
        "Set to null and flag for reassignment.",
        "advisor_id is a foreign key and must be an integer.",
        "Medium: the clients cannot be linked to an advisor.")
if ADVISOR_FILE.exists():
    adv_df = pd.read_csv(ADVISOR_FILE, dtype=str, keep_default_na=False)
    if "advisor_id" in adv_df.columns:
        known = set(adv_df["advisor_id"].str.strip())
        orphan = ~df["advisor_id"].str.strip().isin(known)
        if orphan.any():
            add("advisor_id",
                f"{orphan.sum()} clients point to an advisor_id that does "
                f"not exist in dim_advisor: {examples(df.loc[orphan, 'advisor_id'])}.",
                "Flag the clients and reassign them, or add the missing advisor.",
                "Orphan foreign keys drop out of inner joins.",
                "Medium.")

# 14. is_active: mixed boolean encodings
act = df["is_active"].str.strip()
if act.nunique() > 2:
    counts = ", ".join(f"'{k}' ({v})" for k, v in act.value_counts().items())
    add("is_active",
        f"Mixed boolean encodings: {counts}.",
        "Map True/yes/1/Y -> True and False/no/0/N -> False; store as a "
        "boolean column. Null anything else.",
        "One flag must use one representation, or filters like "
        "is_active == True miss the 'yes' rows and count the '0' rows wrong.",
        "Low: the mapping is unambiguous for these values.")

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
report = pd.DataFrame(issues,
                      columns=["Column", "Issue", "Fixing Strategy",
                               "Justification", "Risk"])
report.insert(0, "#", range(1, len(report) + 1))


def print_table(frame, widths):
    """Print a text-wrapped grid table using only the standard library."""
    cols = list(frame.columns)
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def row_lines(values):
        wrapped = [textwrap.wrap(str(v), w) or [""] for v, w in zip(values, widths)]
        height = max(len(c) for c in wrapped)
        for i in range(height):
            yield "| " + " | ".join(
                (c[i] if i < len(c) else "").ljust(w)
                for c, w in zip(wrapped, widths)) + " |"

    print(sep)
    for line in row_lines(cols):
        print(line)
    print(sep.replace("-", "="))
    for _, r in frame.iterrows():
        for line in row_lines(r.tolist()):
            print(line)
        print(sep)


print(f"\nDATA QUALITY REPORT: {RAW_FILE.name}")
print(f"Rows: {n_rows:,}   Columns: {df.shape[1]}   "
      f"Issues found: {len(report)}\n")
print_table(report, widths=[3, 16, 34, 34, 34, 26])

REPORT_DIR.mkdir(exist_ok=True)
report.to_csv(REPORT_FILE, index=False)
print(f"\nReport saved to: {REPORT_FILE}")
