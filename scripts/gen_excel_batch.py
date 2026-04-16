import json, os

OUT = "/home/ubuntu/PycharmProjects/DocWain/finetune_artifacts/sprint/claude_quality/excel_csv_batch.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

SYS = "You are DocWain, an enterprise document intelligence assistant. You analyse documents with deep contextual understanding, extract structured information, identify patterns and anomalies, and provide holistic analysis grounded in evidence. You reason step-by-step before answering, state your confidence level, and cite specific sources. When information is insufficient, you say so clearly rather than guessing."

def make(query, reasoning, answer):
    text = (
        f"<|im_start|>system\n{SYS}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>"
    )
    return {"text": text, "category": "excel_csv", "difficulty": "hard", "source": "claude_quality"}

examples = []

# ── MULTI-SHEET AGGREGATION (4) ──────────────────────────────────────────────

examples.append(make(
    query="""I have a workbook with three regional sales sheets:

Sheet "North": Q1=142000, Q2=158000, Q3=171000, Q4=189000
Sheet "South": Q1=98000,  Q2=112000, Q3=105000, Q4=131000
Sheet "West":  Q1=210000, Q2=198000, Q3=225000, Q4=243000

Aggregate total annual revenue by region and company-wide, identify the highest-performing quarter overall, and rank regions by full-year contribution.""",
    reasoning="""Step 1 — Annual totals per region:
  North: 142000+158000+171000+189000 = 660000
  South: 98000+112000+105000+131000  = 446000
  West:  210000+198000+225000+243000 = 876000

Step 2 — Company-wide total: 660000+446000+876000 = 1,982,000

Step 3 — Quarter totals (all regions combined):
  Q1: 142000+98000+210000  = 450000
  Q2: 158000+112000+198000 = 468000
  Q3: 171000+105000+225000 = 501000
  Q4: 189000+131000+243000 = 563000
  Highest quarter: Q4 at 563,000.

Step 4 — Region share:
  West:  876000/1982000 = 44.2%
  North: 660000/1982000 = 33.3%
  South: 446000/1982000 = 22.5%

Step 5 — QoQ growth check: every region grew Q3→Q4 (North +10.5%, South +24.8%, West +8.0%), confirming broad Q4 strength.""",
    answer="""## Multi-Sheet Revenue Aggregation

### Annual Totals by Region

| Region | Q1 | Q2 | Q3 | Q4 | Full Year | Share |
|--------|----|----|----|----|-----------|-------|
| North  | $142,000 | $158,000 | $171,000 | $189,000 | **$660,000** | 33.3% |
| South  | $98,000  | $112,000 | $105,000 | $131,000 | **$446,000** | 22.5% |
| West   | $210,000 | $198,000 | $225,000 | $243,000 | **$876,000** | 44.2% |
| **Total** | **$450,000** | **$468,000** | **$501,000** | **$563,000** | **$1,982,000** | 100% |

### Key Findings

1. **Highest-performing quarter:** Q4 at $563,000 — driven by strong closes in all three regions.
2. **Region rank:** West (1st, 44.2%) › North (2nd, 33.3%) › South (3rd, 22.5%).
3. **South anomaly:** Q3 dipped to $105,000 (−6.3% vs Q2) before recovering in Q4. Warrants investigation.
4. **West Q2 dip:** $198,000 vs Q1 $210,000 (−5.7%) is the only other intra-year decline.

### Recommended Formula (Excel)
```
=SUM(North:West!B2)   // cross-sheet Q1 sum (assuming B2 holds Q1 in each sheet)
```

**Confidence: High** — all arithmetic verified; no missing values in source data."""
))

examples.append(make(
    query="""Workbook has sheets: "Headcount_2023", "Headcount_2024", "Salary_Bands".
Headcount_2023: Eng=45, Sales=30, Ops=20, HR=10 (total 105)
Headcount_2024: Eng=58, Sales=35, Ops=18, Ops=18, HR=12 (total 123)
Salary_Bands (annual avg): Eng=$115k, Sales=$85k, Ops=$65k, HR=$72k

Calculate total payroll cost for both years and the incremental cost from headcount changes.""",
    reasoning="""Step 1 — 2023 payroll:
  Eng: 45×115000 = 5,175,000
  Sales: 30×85000  = 2,550,000
  Ops: 20×65000    = 1,300,000
  HR: 10×72000     = 720,000
  Total 2023 = 9,745,000

Step 2 — 2024 payroll:
  Eng: 58×115000 = 6,670,000
  Sales: 35×85000  = 2,975,000
  Ops: 18×65000    = 1,170,000
  HR: 12×72000     = 864,000
  Total 2024 = 11,679,000

Step 3 — Incremental cost = 11,679,000 − 9,745,000 = 1,934,000

Step 4 — Incremental by dept:
  Eng: +13 heads × 115k = +1,495,000 (77.3% of increment)
  Sales: +5 × 85k       = +425,000
  Ops: −2 × 65k         = −130,000
  HR: +2 × 72k          = +144,000

Step 5 — Salary bands are averages; actual cost may vary ±10% due to seniority mix.""",
    answer="""## Cross-Sheet Payroll Aggregation (2023–2024)

### Payroll by Department

| Dept | HC 2023 | HC 2024 | Δ HC | 2023 Cost | 2024 Cost | Δ Cost |
|------|---------|---------|------|-----------|-----------|--------|
| Engineering | 45 | 58 | +13 | $5,175,000 | $6,670,000 | +$1,495,000 |
| Sales | 30 | 35 | +5 | $2,550,000 | $2,975,000 | +$425,000 |
| Operations | 20 | 18 | −2 | $1,300,000 | $1,170,000 | −$130,000 |
| HR | 10 | 12 | +2 | $720,000 | $864,000 | +$144,000 |
| **Total** | **105** | **123** | **+18** | **$9,745,000** | **$11,679,000** | **+$1,934,000** |

### Key Findings

1. **Total incremental payroll cost: +$1,934,000 (+19.8% YoY).**
2. Engineering dominates the increment — 77.3% of the increase stems from 13 new Eng hires.
3. Ops headcount reduction saved $130,000, partially offsetting growth costs.
4. Headcount grew 17.1% (105→123) while payroll grew 19.8%, indicating the new hires skew toward higher-band roles (Engineering).

**Confidence: High** — calculations use provided band averages; caveat that actual payroll depends on individual salaries within bands."""
))

examples.append(make(
    query="""Three product sheets (Electronics, Apparel, Grocery) each have columns: Product, Units_Sold, Unit_Cost, Unit_Price.

Electronics: Laptop(200,800,1200), Phone(500,300,550), Tablet(150,250,420)
Apparel: Jacket(300,40,90), Jeans(600,25,60), Shoes(400,35,80)
Grocery: Coffee(1000,5,12), Tea(800,3,8), Snacks(1500,2,5)

Aggregate gross profit by sheet and overall. Which sheet has the best margin?""",
    reasoning="""Gross Profit = (Unit_Price − Unit_Cost) × Units_Sold

Electronics:
  Laptop: (1200−800)×200 = 400×200 = 80,000
  Phone:  (550−300)×500  = 250×500 = 125,000
  Tablet: (420−250)×150  = 170×150 = 25,500
  Sheet GP = 230,500; Revenue = (1200×200)+(550×500)+(420×150) = 240000+275000+63000 = 578,000
  Margin = 230500/578000 = 39.9%

Apparel:
  Jacket: (90−40)×300  = 50×300  = 15,000
  Jeans:  (60−25)×600  = 35×600  = 21,000
  Shoes:  (80−35)×400  = 45×400  = 18,000
  Sheet GP = 54,000; Revenue = 27000+36000+32000 = 95,000
  Margin = 54000/95000 = 56.8%

Grocery:
  Coffee: (12−5)×1000  = 7×1000  = 7,000
  Tea:    (8−3)×800    = 5×800   = 4,000
  Snacks: (5−2)×1500   = 3×1500  = 4,500
  Sheet GP = 15,500; Revenue = 12000+6400+7500 = 25,900
  Margin = 15500/25900 = 59.8%

Overall GP = 230500+54000+15500 = 300,000
Overall Revenue = 578000+95000+25900 = 698,900
Overall margin = 300000/698900 = 42.9%""",
    answer="""## Multi-Sheet Gross Profit Aggregation

### By Product

| Sheet | Product | Revenue | Gross Profit | Margin |
|-------|---------|---------|--------------|--------|
| Electronics | Laptop | $240,000 | $80,000 | 33.3% |
| Electronics | Phone | $275,000 | $125,000 | 45.5% |
| Electronics | Tablet | $63,000 | $25,500 | 40.5% |
| Apparel | Jacket | $27,000 | $15,000 | 55.6% |
| Apparel | Jeans | $36,000 | $21,000 | 58.3% |
| Apparel | Shoes | $32,000 | $18,000 | 56.3% |
| Grocery | Coffee | $12,000 | $7,000 | 58.3% |
| Grocery | Tea | $6,400 | $4,000 | 62.5% |
| Grocery | Snacks | $7,500 | $4,500 | 60.0% |

### Sheet Summary

| Sheet | Revenue | Gross Profit | **Margin** |
|-------|---------|--------------|------------|
| Electronics | $578,000 | $230,500 | 39.9% |
| Apparel | $95,000 | $54,000 | 56.8% |
| **Grocery** | $25,900 | $15,500 | **59.8%** |
| **Overall** | **$698,900** | **$300,000** | **42.9%** |

### Key Findings

1. **Best margin: Grocery at 59.8%**, despite lowest absolute revenue.
2. Electronics contributes 76.8% of total revenue but only 33-46% margins.
3. Apparel (56.8%) and Grocery (59.8%) are both high-margin relative to Electronics.
4. Strategy implication: Electronics drives volume, Grocery/Apparel drive profitability per dollar.

**Confidence: High** — all values computed directly from provided data."""
))

examples.append(make(
    query="""Workbook: "Budget" sheet has planned spend by dept; "Actuals_Q1" and "Actuals_Q2" have real spend.

Budget (annual): IT=500k, Marketing=300k, HR=150k, Ops=200k
Actuals_Q1: IT=140k, Marketing=95k, HR=38k, Ops=52k
Actuals_Q2: IT=155k, Marketing=110k, HR=40k, Ops=49k

Calculate H1 actuals, annualised run-rate, variance to budget, and flag departments at risk of overspend.""",
    reasoning="""Step 1 — H1 actuals (Q1+Q2):
  IT: 140+155=295k, Mktg: 95+110=205k, HR: 38+40=78k, Ops: 52+49=101k
  Total H1: 679k

Step 2 — Annualised run-rate = H1 × 2:
  IT: 590k, Mktg: 410k, HR: 156k, Ops: 202k, Total: 1,358k

Step 3 — Annual budget total: 500+300+150+200 = 1,150k

Step 4 — Variance (Budget − Run-rate); negative = over budget:
  IT: 500−590 = −90k (−18.0%)
  Mktg: 300−410 = −110k (−36.7%)
  HR: 150−156 = −6k (−4.0%)
  Ops: 200−202 = −2k (−1.0%)

Step 5 — Risk flags: IT and Marketing are significantly over annualised run-rate. HR and Ops are marginally over but within noise.""",
    answer="""## Budget vs Actuals — H1 Multi-Sheet Analysis

### Spend Summary

| Dept | Budget (Annual) | H1 Actual | Run-Rate (×2) | Variance | % Over/Under |
|------|----------------|-----------|---------------|----------|--------------|
| IT | $500k | $295k | $590k | **−$90k** | −18.0% |
| Marketing | $300k | $205k | $410k | **−$110k** | −36.7% |
| HR | $150k | $78k | $156k | −$6k | −4.0% |
| Operations | $200k | $101k | $202k | −$2k | −1.0% |
| **Total** | **$1,150k** | **$679k** | **$1,358k** | **−$208k** | **−18.1%** |

### Risk Assessment

| Dept | Status | Action |
|------|--------|--------|
| Marketing | 🔴 HIGH RISK | 36.7% over run-rate — immediate review required |
| IT | 🔴 HIGH RISK | 18.0% over — infrastructure or licence cost spike likely |
| HR | 🟡 WATCH | Marginally over; monitor Q3 headcount additions |
| Operations | 🟢 ON TRACK | Within 1% — no action needed |

### Notes
- Run-rate assumes H1 pace continues; actual H2 may differ due to seasonal patterns.
- Marketing overspend warrants campaign-level drill-down (Q2 alone: $110k vs $75k H1-prorated budget).

**Confidence: High** — arithmetic verified; caveats noted for run-rate linearity assumption."""
))

# ── FORMULA INTERPRETATION (4) ──────────────────────────────────────────────

examples.append(make(
    query="""Explain what this Excel formula does and what result it would produce given the data:

Formula (in cell F2): =IFERROR(VLOOKUP(E2, $B$2:$D$10, 3, FALSE) * C2 * (1 - $G$1), "Not Found")

Context: E2="SKU-204", Column B has SKU codes, Column D has unit prices, C2=50 (quantity), G1=0.15 (discount rate). SKU-204 is in B7 with D7=120.""",
    reasoning="""Step 1 — Parse the formula structure:
  Outer: IFERROR(..., "Not Found") — catches any lookup error.
  Inner: VLOOKUP(E2, $B$2:$D$10, 3, FALSE)
    lookup_value = E2 = "SKU-204"
    table_array = $B$2:$D$10 (columns B, C, D — 3 columns)
    col_index = 3 → returns column D (unit price)
    range_lookup = FALSE → exact match only

Step 2 — VLOOKUP result: SKU-204 found at B7, D7=120 → returns 120.

Step 3 — Full expression: 120 × C2 × (1 − $G$1)
  = 120 × 50 × (1 − 0.15)
  = 120 × 50 × 0.85
  = 6000 × 0.85
  = 5,100

Step 4 — IFERROR: no error occurred, so result = 5,100.

Step 5 — Business meaning: discounted revenue for 50 units of SKU-204 at $120 each with 15% discount.""",
    answer="""## Formula Interpretation: Discounted Revenue Lookup

### Formula Breakdown

```
=IFERROR(VLOOKUP(E2, $B$2:$D$10, 3, FALSE) * C2 * (1 - $G$1), "Not Found")
```

| Component | Purpose | Value |
|-----------|---------|-------|
| `VLOOKUP(E2, $B$2:$D$10, 3, FALSE)` | Looks up SKU-204 in col B, returns col D (unit price) | **120** |
| `* C2` | Multiplies by quantity in C2 | × 50 |
| `* (1 - $G$1)` | Applies discount: 1 − 0.15 = 0.85 | × 0.85 |
| `IFERROR(..., "Not Found")` | Returns "Not Found" if SKU missing | No error |

### Calculation

```
120 × 50 × 0.85 = 5,100
```

**Cell F2 result: $5,100**

### Design Notes

- `$B$2:$D$10` uses absolute references — formula can be copied down column F without the lookup range drifting.
- `$G$1` is also absolute — a single discount cell controls all rows; changing G1 updates every line instantly.
- `FALSE` (exact match) is correct for SKU lookups; using TRUE would risk wrong-SKU matches on sorted data.
- **Fragility:** If column C or D is inserted between B and D, col_index=3 would silently return the wrong column. Use `MATCH` inside `INDEX` for robustness in production sheets.

**Confidence: High** — formula logic and arithmetic fully verified."""
))

examples.append(make(
    query="""What does this formula array-calculate, and what is the result?

=SUMPRODUCT((A2:A20="West")*(B2:B20="Q3")*(C2:C20))

Data excerpt (relevant rows): Row 5: West, Q3, 45000 | Row 9: West, Q1, 32000 | Row 14: West, Q3, 28000 | Row 18: East, Q3, 51000. All other rows are different region/quarter combos.""",
    reasoning="""Step 1 — SUMPRODUCT evaluates element-wise multiplication across three arrays of length 19.

Step 2 — For each row, compute three Boolean/value factors:
  (A=West): 1 if West, else 0
  (B=Q3):   1 if Q3, else 0
  (C):      the numeric value in col C

Step 3 — Row-by-row non-zero contributions:
  Row 5:  West(1) × Q3(1)  × 45000 = 45000
  Row 9:  West(1) × Q1(0)  × 32000 = 0  (Q1 not Q3)
  Row 14: West(1) × Q3(1)  × 28000 = 28000
  Row 18: East(0) × Q3(1)  × 51000 = 0  (East not West)
  All others = 0 by definition.

Step 4 — SUMPRODUCT sums all products: 45000 + 0 + 28000 + 0 = 73,000.

Step 5 — This is equivalent to a SUMIFS: =SUMIFS(C2:C20, A2:A20, "West", B2:B20, "Q3").""",
    answer="""## SUMPRODUCT Formula Interpretation

### Formula

```
=SUMPRODUCT((A2:A20="West")*(B2:B20="Q3")*(C2:C20))
```

### How It Works

This is a **multi-condition conditional sum** — an alternative to `SUMIFS`.

1. `(A2:A20="West")` → array of 1s and 0s (1 where region is West)
2. `(B2:B20="Q3")` → array of 1s and 0s (1 where quarter is Q3)
3. `(C2:C20)` → array of numeric values
4. Element-wise multiplication: only rows where **both** conditions are true (1×1=1) contribute their C value; all others multiply to 0.
5. `SUMPRODUCT` sums the resulting array.

### Row-Level Evaluation (relevant rows)

| Row | Region | Quarter | Value | A=West | B=Q3 | Contribution |
|-----|--------|---------|-------|--------|------|--------------|
| 5 | West | Q3 | 45,000 | 1 | 1 | **45,000** |
| 9 | West | Q1 | 32,000 | 1 | 0 | 0 |
| 14 | West | Q3 | 28,000 | 1 | 1 | **28,000** |
| 18 | East | Q3 | 51,000 | 0 | 1 | 0 |

**Result: 73,000**

### Equivalent Formula

```excel
=SUMIFS(C2:C20, A2:A20, "West", B2:B20, "Q3")   // identical result, often faster on large ranges
```

**Confidence: High** — logic verified; result confirmed by manual trace."""
))

examples.append(make(
    query="""Interpret this formula and describe all edge cases:

=IF(AND(D2>0, E2<>""), ROUND(D2/E2*100, 2), IF(E2="", "Missing Denominator", "Zero or Negative"))

Where D2 = Net Profit, E2 = Revenue. Test with: (a) D2=15000, E2=200000; (b) D2=0, E2=150000; (c) D2=5000, E2="" ; (d) D2=-3000, E2=100000.""",
    reasoning="""The outer IF: AND(D2>0, E2<>"") — both D2 strictly positive AND E2 not empty.
  If TRUE → ROUND(D2/E2*100, 2)   [profit margin %]
  If FALSE → nested IF:
    If E2="" → "Missing Denominator"
    Else → "Zero or Negative"

Case (a): D2=15000>0 ✓, E2=200000≠"" ✓ → AND=TRUE
  ROUND(15000/200000×100, 2) = ROUND(7.5, 2) = 7.50

Case (b): D2=0, NOT >0 → AND=FALSE; E2≠"" → "Zero or Negative"
  Returns "Zero or Negative"

Case (c): D2=5000>0 ✓ but E2="" → AND=FALSE; E2="" → "Missing Denominator"
  Returns "Missing Denominator"

Case (d): D2=−3000, NOT >0 → AND=FALSE; E2=100000≠"" → "Zero or Negative"
  Returns "Zero or Negative" (cannot distinguish zero from negative without further nesting)""",
    answer="""## Formula Interpretation: Profit Margin with Error Handling

### Formula Logic

```excel
=IF(AND(D2>0, E2<>""),
    ROUND(D2/E2*100, 2),
    IF(E2="", "Missing Denominator", "Zero or Negative"))
```

**Purpose:** Calculates net profit margin (%) only when both inputs are valid; otherwise returns a descriptive error string.

### Decision Tree

```
D2 > 0  AND  E2 ≠ ""
        │
       YES → ROUND(D2/E2×100, 2)   ← profit margin %
        │
        NO → E2 = ""?
                │
               YES → "Missing Denominator"
                │
                NO → "Zero or Negative"
```

### Test Cases

| Case | D2 | E2 | AND result | Output |
|------|----|----|------------|--------|
| (a) | 15,000 | 200,000 | TRUE | **7.50** (%) |
| (b) | 0 | 150,000 | FALSE (D2 not >0) | **"Zero or Negative"** |
| (c) | 5,000 | "" | FALSE (E2 empty) | **"Missing Denominator"** |
| (d) | −3,000 | 100,000 | FALSE (D2 not >0) | **"Zero or Negative"** |

### Edge Case Issues

1. **Cases (b) and (d) return identical strings** — the formula cannot distinguish zero profit from a loss. Fix: add a third branch `IF(D2=0, "Break Even", "Loss")`.
2. **E2=0 (numeric zero):** E2<>"" is TRUE, AND may be TRUE if D2>0, causing a #DIV/0! error. Fix: add `E2<>0` to the AND condition.
3. **ROUND to 2dp** is appropriate for percentage display; omit if downstream calculations need full precision.

**Confidence: High** — all four cases traced through the logic tree."""
))

examples.append(make(
    query="""Explain this formula used for running balance in a cash-flow sheet:

Cell C2: =B2
Cell C3 (copied down to C50): =IF(B3<>"", C2+B3, C2)

Column B holds daily cash movements (positive=inflow, negative=outflow). B3=+5000, B4=-2000, B5="" (blank), B6=+8000. What are C2–C6 if C2 starts from B2=10000?""",
    reasoning="""C2 = B2 = 10,000 (seed/opening balance)

C3: B3=+5000 ≠ "" → C2+B3 = 10000+5000 = 15,000
C4: B4=−2000 ≠ "" → C3+B4 = 15000+(−2000) = 13,000
C5: B5="" → C4 (carry forward) = 13,000
C6: B6=+8000 ≠ "" → C5+B6 = 13000+8000 = 21,000

The blank-row logic (C5=C4) means an empty B cell preserves the balance rather than adding zero — useful for visual separators between weeks or months without affecting the running total.""",
    answer="""## Running Balance Formula Interpretation

### Formula Design

```excel
C2: =B2
C3: =IF(B3<>"", C2+B3, C2)   ← copied through C50
```

**Pattern:** Cumulative sum where blank rows are treated as "no transaction" (balance carries forward unchanged), not as zero-value entries.

### Step-by-Step Evaluation

| Row | B (Movement) | Formula in C | Calculation | **C (Balance)** |
|-----|-------------|--------------|-------------|-----------------|
| 2 | +10,000 | =B2 | seed | **$10,000** |
| 3 | +5,000 | B3≠"" → C2+B3 | 10,000+5,000 | **$15,000** |
| 4 | −2,000 | B4≠"" → C3+B4 | 15,000−2,000 | **$13,000** |
| 5 | *(blank)* | B5="" → C4 | carry forward | **$13,000** |
| 6 | +8,000 | B6≠"" → C5+B6 | 13,000+8,000 | **$21,000** |

### Design Assessment

**Strengths:**
- Blank rows act as visual separators (e.g., week breaks) without corrupting the balance.
- Simple copy-down pattern — no absolute references needed.

**Weaknesses / Risks:**
- If B contains a formula that returns `""` on purpose (e.g., `=IF(condition,"",value)`), that row will be silently skipped even if it should add zero.
- No protection against non-numeric B entries (text would cause a #VALUE! error).
- Improved version: `=IF(ISNUMBER(B3), C2+B3, C2)` catches both blanks and text safely.

**Confidence: High** — trace verified against provided data."""
))

# ── TREND ANALYSIS (4) ──────────────────────────────────────────────────────

examples.append(make(
    query="""Monthly revenue data for 12 months:
Jan=82k, Feb=78k, Mar=91k, Apr=95k, May=99k, Jun=107k,
Jul=103k, Aug=110k, Sep=115k, Oct=121k, Nov=118k, Dec=130k

Calculate MoM growth rates, 3-month moving average, identify the trend, and project January next year.""",
    reasoning="""MoM growth = (current − prior)/prior × 100:
  Feb: (78−82)/82 = −4.9%
  Mar: (91−78)/78 = +16.7%
  Apr: (95−91)/91 = +4.4%
  May: (99−95)/95 = +4.2%
  Jun: (107−99)/99 = +8.1%
  Jul: (103−107)/107 = −3.7%
  Aug: (110−103)/103 = +6.8%
  Sep: (115−110)/110 = +4.5%
  Oct: (121−115)/115 = +5.2%
  Nov: (118−121)/121 = −2.5%
  Dec: (130−118)/118 = +10.2%

3-month moving averages (centred on month 2 of each window):
  Feb avg (Jan-Mar): (82+78+91)/3 = 83.7k
  Mar avg (Feb-Apr): (78+91+95)/3 = 88.0k
  Apr avg (Mar-May): (91+95+99)/3 = 95.0k
  May avg (Apr-Jun): (95+99+107)/3 = 100.3k
  Jun avg (May-Jul): (99+107+103)/3 = 103.0k
  Jul avg (Jun-Aug): (107+103+110)/3 = 106.7k
  Aug avg (Jul-Sep): (103+110+115)/3 = 109.3k
  Sep avg (Aug-Oct): (110+115+121)/3 = 115.3k
  Oct avg (Sep-Nov): (115+121+118)/3 = 118.0k
  Nov avg (Oct-Dec): (121+118+130)/3 = 123.0k

Trend: Despite 3 negative MoM months, the moving average is monotonically increasing. Jan→Dec raw growth = (130−82)/82 = 58.5% in 11 months.

Linear projection: average MoM growth (excluding dips) ≈ +4.5%. Dec=130k × 1.045 ≈ 135.9k for Jan next year.""",
    answer="""## Monthly Revenue Trend Analysis

### Raw Data with MoM Growth

| Month | Revenue | MoM Growth | 3-Month MA |
|-------|---------|-----------|------------|
| Jan | $82k | — | — |
| Feb | $78k | −4.9% | $83.7k |
| Mar | $91k | +16.7% | $88.0k |
| Apr | $95k | +4.4% | $95.0k |
| May | $99k | +4.2% | $100.3k |
| Jun | $107k | +8.1% | $103.0k |
| Jul | $103k | −3.7% | $106.7k |
| Aug | $110k | +6.8% | $109.3k |
| Sep | $115k | +4.5% | $115.3k |
| Oct | $121k | +5.2% | $118.0k |
| Nov | $118k | −2.5% | $123.0k |
| Dec | $130k | +10.2% | — |

### Trend Findings

1. **Overall trajectory: strong upward.** Revenue grew 58.5% from Jan to Dec.
2. **Three pullback months** (Feb, Jul, Nov) followed immediately by strong recoveries — characteristic of a healthy growth trend with demand seasonality, not structural decline.
3. **3-month MA** rises consistently from $83.7k to $123.0k, confirming the trend is real, not noise.
4. **Acceleration in Dec** (+10.2%) may reflect year-end deals or seasonal uplift.

### January Projection

Using average forward MoM rate from the last 6 months (+4.8%):
```
$130k × 1.048 ≈ $136.2k
```
Conservative range: **$133k–$139k**

**Confidence: Medium-High** — linear projection assumes continuation of recent momentum; actual results depend on seasonality and pipeline."""
))

examples.append(make(
    query="""Quarterly website traffic and conversion data over 2 years:

Year 1: Q1(120k visits, 2.1%), Q2(135k, 2.3%), Q3(128k, 2.0%), Q4(155k, 2.5%)
Year 2: Q1(142k, 2.4%), Q2(161k, 2.6%), Q3(152k, 2.2%), Q4(189k, 2.8%)

Analyse traffic trends, conversion trends, calculate converted visitors, and identify seasonality patterns.""",
    reasoning="""Converted visitors = visits × conversion rate:
  Y1Q1: 120000×0.021=2520, Y1Q2: 135000×0.023=3105, Y1Q3: 128000×0.020=2560, Y1Q4: 155000×0.025=3875
  Y2Q1: 142000×0.024=3408, Y2Q2: 161000×0.026=4186, Y2Q3: 152000×0.022=3344, Y2Q4: 189000×0.028=5292

YoY traffic growth by quarter:
  Q1: (142−120)/120 = +18.3%
  Q2: (161−135)/135 = +19.3%
  Q3: (152−128)/128 = +18.8%
  Q4: (189−155)/155 = +21.9%

YoY conversion rate change by quarter:
  Q1: 2.4−2.1=+0.3pp, Q2: 2.6−2.3=+0.3pp, Q3: 2.2−2.0=+0.2pp, Q4: 2.8−2.5=+0.3pp

Seasonality: Q3 is consistently the lowest traffic AND conversion quarter both years (mid-summer dip). Q4 is consistently highest. Pattern repeats exactly — strong seasonality signal.

YoY converted visitors growth:
  Q1: (3408−2520)/2520=+35.2%  Q2: (4186−3105)/3105=+34.8%
  Q3: (3344−2560)/2560=+30.6%  Q4: (5292−3875)/3875=+36.6%""",
    answer="""## Website Traffic & Conversion Trend Analysis

### Full Dataset with Converted Visitors

| Period | Visits | Conv Rate | Converted | YoY Visit Δ | YoY Conv Δ |
|--------|--------|-----------|-----------|------------|------------|
| Y1 Q1 | 120,000 | 2.1% | 2,520 | — | — |
| Y1 Q2 | 135,000 | 2.3% | 3,105 | — | — |
| Y1 Q3 | 128,000 | 2.0% | 2,560 | — | — |
| Y1 Q4 | 155,000 | 2.5% | 3,875 | — | — |
| Y2 Q1 | 142,000 | 2.4% | 3,408 | +18.3% | +0.3pp |
| Y2 Q2 | 161,000 | 2.6% | 4,186 | +19.3% | +0.3pp |
| Y2 Q3 | 152,000 | 2.2% | 3,344 | +18.8% | +0.2pp |
| Y2 Q4 | 189,000 | 2.8% | 5,292 | +21.9% | +0.3pp |

### Key Trends

1. **Traffic growing ~19% YoY across all quarters** — consistent, not lumpy.
2. **Conversion rate improving +0.2–0.3pp YoY** — site optimisation or better traffic quality.
3. **Converted visitors grew 31–37% YoY** — multiplicative benefit of both improvements.
4. **Q4 accelerating most** (+21.9% traffic, +36.6% conversions) — likely seasonal campaign uplift.

### Seasonality Pattern (repeatable both years)

```
Q4 > Q2 > Q1 > Q3   (both visits and conversion rate)
```
Q3 is the consistent trough — summer lull. Plan reduced ad spend and maintenance windows in Q3; increase budget for Q4 campaigns.

**Confidence: High** — two full years confirms seasonality; projecting Y3 Q4 ≈ 189k × 1.20 ≈ 227k visits at ~3.1% ≈ 7,000 conversions."""
))

examples.append(make(
    query="""Employee attrition data across 6 quarters:
Q1Y1: headcount=500, leavers=12
Q2Y1: HC=495, leavers=8
Q3Y1: HC=490, leavers=15
Q4Y1: HC=482, leavers=7
Q1Y2: HC=510, leavers=18
Q2Y2: HC=498, leavers=22

Calculate annualised attrition rate per quarter, trend, and flag if the trend is concerning.""",
    reasoning="""Annualised attrition = (leavers/HC) × 4 (quarters to annual):
  Q1Y1: (12/500)×4 = 0.024×4 = 9.6%
  Q2Y1: (8/495)×4  = 0.01616×4 = 6.5%
  Q3Y1: (15/490)×4 = 0.03061×4 = 12.2%
  Q4Y1: (7/482)×4  = 0.01452×4 = 5.8%
  Q1Y2: (18/510)×4 = 0.03529×4 = 14.1%
  Q2Y2: (22/498)×4 = 0.04418×4 = 17.7%

Industry benchmark for knowledge workers: ~10–12% annual attrition is normal.
Q1Y2 (14.1%) and Q2Y2 (17.7%) both exceed this.

Trend direction: Q4Y1 was the lowest (5.8%) — possible retention initiative. But Q1Y2 and Q2Y2 show sharp acceleration: 5.8% → 14.1% → 17.7% over three quarters. 17.7% annualised is significantly elevated.

Leavers in Q2Y2 = 22 — the highest absolute count in the dataset, despite HC being lower than Q1Y2.""",
    answer="""## Employee Attrition Trend Analysis

### Quarterly Attrition (Annualised)

| Quarter | Headcount | Leavers | Annualised Rate | vs. 10% Benchmark |
|---------|-----------|---------|-----------------|-------------------|
| Q1 Y1 | 500 | 12 | 9.6% | Within range |
| Q2 Y1 | 495 | 8 | 6.5% | Below — good |
| Q3 Y1 | 490 | 15 | 12.2% | Slightly elevated |
| Q4 Y1 | 482 | 7 | 5.8% | Low — very good |
| Q1 Y2 | 510 | 18 | 14.1% | ⚠ Elevated |
| Q2 Y2 | 498 | 22 | **17.7%** | 🔴 **Concerning** |

### Trend Assessment

**Verdict: Concerning — immediate investigation warranted.**

1. **Acceleration is sharp and sustained:** Q4Y1 (5.8%) → Q1Y2 (14.1%) → Q2Y2 (17.7%) — a tripling of the annualised rate in two quarters.
2. **Q2Y2 absolute leavers (22) is the dataset maximum**, despite headcount being below the Q1Y2 peak — the rate is genuine, not a denominator effect.
3. **The Q4Y1 low** may reflect a retention initiative whose effect has now worn off, or end-of-year bonus lock-in.
4. **Net headcount change:** Started Y1 at 500, now at 498 after hiring in Y2 — hiring is barely keeping pace with departures.

### Recommended Actions

- Exit interview analysis for Q1–Q2 Y2 cohort (N=40 leavers).
- Segment by department and tenure — is attrition concentrated or broad?
- If rate continues at Q2Y2 pace: projected Y2 annual = ~17.7% × 498 ≈ 88 leavers.

**Confidence: High** — calculations verified; benchmark reference is industry standard."""
))

examples.append(make(
    query="""Product defect rate data from manufacturing line over 8 weeks:
Wk1=1.2%, Wk2=1.4%, Wk3=1.1%, Wk4=0.9%, Wk5=1.0%, Wk6=1.3%, Wk7=2.1%, Wk8=2.8%

Calculate the mean, standard deviation, control limits (±2σ), identify out-of-control points, and describe the process trend.""",
    reasoning="""Mean = (1.2+1.4+1.1+0.9+1.0+1.3+2.1+2.8)/8
  = 11.8/8 = 1.475%

Variance: deviations from mean:
  Wk1: 1.2−1.475=−0.275 → sq=0.07563
  Wk2: 1.4−1.475=−0.075 → sq=0.00563
  Wk3: 1.1−1.475=−0.375 → sq=0.14063
  Wk4: 0.9−1.475=−0.575 → sq=0.33063
  Wk5: 1.0−1.475=−0.475 → sq=0.22563
  Wk6: 1.3−1.475=−0.175 → sq=0.03063
  Wk7: 2.1−1.475=+0.625 → sq=0.39063
  Wk8: 2.8−1.475=+1.325 → sq=1.75563
  Sum of sq = 2.95501
  Variance (population) = 2.95501/8 = 0.36938
  σ = √0.36938 = 0.6078%

Control limits (±2σ):
  UCL = 1.475 + 2×0.6078 = 1.475 + 1.216 = 2.691%
  LCL = 1.475 − 2×0.6078 = 1.475 − 1.216 = 0.259% (floored at 0)

Out-of-control: Wk8 = 2.8% > UCL 2.691% → beyond control limit.
Wk7 = 2.1% is within limits but trending sharply upward.""",
    answer="""## Manufacturing Defect Rate — SPC Analysis

### Weekly Data

| Week | Defect Rate | Deviation from Mean | In Control? |
|------|-------------|--------------------:|-------------|
| 1 | 1.20% | −0.275pp | ✓ |
| 2 | 1.40% | −0.075pp | ✓ |
| 3 | 1.10% | −0.375pp | ✓ |
| 4 | 0.90% | −0.575pp | ✓ |
| 5 | 1.00% | −0.475pp | ✓ |
| 6 | 1.30% | −0.175pp | ✓ |
| 7 | 2.10% | +0.625pp | ✓ (near) |
| **8** | **2.80%** | +1.325pp | **🔴 OUT** |

### Control Chart Parameters

| Statistic | Value |
|-----------|-------|
| Mean (x̄) | **1.475%** |
| Std Dev (σ) | **0.608%** |
| UCL (+2σ) | **2.691%** |
| LCL (−2σ) | 0.259% (effective: 0%) |

### Process Assessment

1. **Week 8 (2.8%) breaches the UCL (2.691%)** — statistically out of control. A special cause is likely present.
2. **Weeks 4–5 were the best period** (0.9%, 1.0%) — process was improving. Wk6 started an uptick.
3. **Wks 6→7→8 show an accelerating upward run** — three consecutive increases. Even without Wk8 breaching UCL, this run pattern (Nelson Rule) signals a non-random shift.
4. **Possible causes:** tooling wear, material batch change, operator shift rotation, or equipment calibration drift starting around Wk6.

### Recommendation

Stop line for root-cause analysis. Investigate changes introduced between Wk5 and Wk6. Do not recalculate control limits until the assignable cause is identified and corrected.

**Confidence: High** — SPC calculations verified; σ uses population formula (N=8)."""
))

# ── ANOMALY DETECTION (4) ────────────────────────────────────────────────────

examples.append(make(
    query="""Expense report data for 12 employees. Monthly totals:
Alice=1200, Bob=980, Carol=1150, Dave=1050, Eve=1080, Frank=990,
Grace=1200, Henry=1020, Iris=1180, Jack=4850, Kim=1100, Leo=1060

Identify anomalies using statistical methods, calculate z-scores, and assess whether any values require investigation.""",
    reasoning="""Step 1 — Mean excluding Jack first to see the baseline:
  Sum of 11 values (excl Jack): 1200+980+1150+1050+1080+990+1200+1020+1180+1100+1060 = 12010
  Mean (11) = 12010/11 = 1091.8

Step 2 — Include all 12 for proper z-score:
  Total = 12010+4850 = 16860; Mean = 16860/12 = 1405

Step 3 — Variance (all 12):
  Deviations squared (using mean=1405):
  Alice: (1200−1405)²=(−205)²=42025
  Bob: (980−1405)²=(−425)²=180625
  Carol: (1150−1405)²=(−255)²=65025
  Dave: (1050−1405)²=(−355)²=126025
  Eve: (1080−1405)²=(−325)²=105625
  Frank: (990−1405)²=(−415)²=172225
  Grace: (1200−1405)²=(−205)²=42025
  Henry: (1020−1405)²=(−385)²=148225
  Iris: (1180−1405)²=(−225)²=50625
  Jack: (4850−1405)²=(3445)²=11868025
  Kim: (1100−1405)²=(−305)²=93025
  Leo: (1060−1405)²=(−345)²=119025
  Sum = 13012500; σ² = 13012500/12 = 1084375; σ = 1041.3

  Jack z-score: (4850−1405)/1041.3 = 3445/1041.3 = 3.31

Step 4 — Without Jack: mean=1091.8, check if any others are anomalous:
  Rough σ of 11 = ~87. Bob (980): z=(980−1091.8)/87≈−1.29 — normal.""",
    answer="""## Expense Report Anomaly Detection

### Dataset Summary

| Employee | Monthly Expenses | Z-Score (full dataset) | Status |
|----------|-----------------|----------------------|--------|
| Alice | $1,200 | −0.20 | Normal |
| Bob | $980 | −0.41 | Normal |
| Carol | $1,150 | −0.24 | Normal |
| Dave | $1,050 | −0.34 | Normal |
| Eve | $1,080 | −0.31 | Normal |
| Frank | $990 | −0.40 | Normal |
| Grace | $1,200 | −0.20 | Normal |
| Henry | $1,020 | −0.37 | Normal |
| Iris | $1,180 | −0.22 | Normal |
| **Jack** | **$4,850** | **+3.31** | 🔴 **ANOMALY** |
| Kim | $1,100 | −0.29 | Normal |
| Leo | $1,060 | −0.33 | Normal |

### Statistical Parameters

| Metric | Value |
|--------|-------|
| Mean (all 12) | $1,405 |
| Std Dev (σ) | $1,041 |
| Z-score threshold (±2σ) | $323 – $2,487 |
| **Jack's deviation from peer mean** | **+$3,758 (+344%)** |

### Findings

1. **Jack's expenses ($4,850) are 3.31 standard deviations above the mean** — a statistically significant outlier. At |z|>3, the probability of this being random variation is <0.27%.
2. **Peer cluster is tight:** excluding Jack, all 11 employees fall within $980–$1,200 (range $220), mean $1,092, σ≈$87. Jack is $3,658 above this peer mean.
3. **All other employees are within ±2σ** — no secondary anomalies.

### Recommended Actions

- Flag Jack's expense report for manager and finance review.
- Request itemised receipts for the full month.
- Check whether Jack had an approved unusual expense (conference, travel, equipment) that was not pre-coded.
- If no approved exceptional items exist, escalate to HR/compliance.

**Confidence: High** — z-score methodology is appropriate for this population size; Jack is unambiguously anomalous."""
))

examples.append(make(
    query="""Daily transaction counts for an e-commerce platform (Mon–Sun over 3 weeks):

Wk1: M=1820, T=1950, W=1880, Th=1910, F=2150, Sa=2400, Su=1600
Wk2: M=1790, T=1930, W=1870, Th=1890, F=2180, Sa=2350, Su=1580
Wk3: M=1810, T=1960, W=420,  Th=1920, F=2200, Sa=2410, Su=1590

Identify anomalies, confirm which are genuine outliers vs. normal weekly patterns, and assess the Wednesday Week 3 figure.""",
    reasoning="""Step 1 — Build day-of-week baselines from Wk1 and Wk2 (before anomaly):
  Mon avg: (1820+1790)/2=1805; Tue: (1950+1930)/2=1940; Wed: (1880+1870)/2=1875
  Thu: (1910+1890)/2=1900; Fri: (2150+2180)/2=2165; Sat: (2400+2350)/2=2375; Sun: (1600+1580)/2=1590

Step 2 — Compare Wk3 actuals to day baselines:
  Mon: 1810 vs 1805 (+0.3%) — normal
  Tue: 1960 vs 1940 (+1.0%) — normal
  Wed: 420  vs 1875 (−77.6%) — extreme anomaly
  Thu: 1920 vs 1900 (+1.1%) — normal
  Fri: 2200 vs 2165 (+1.6%) — normal
  Sat: 2410 vs 2375 (+1.5%) — normal
  Sun: 1590 vs 1590 (0.0%) — normal

Step 3 — Wk3 Wed z-score using all Wed values: values=1880,1870,420; mean=(1880+1870+420)/3=1390; σ=√[((1880-1390)²+(1870-1390)²+(420-1390)²)/3] = √[(490²+480²+970²)/3] = √[(240100+230400+940900)/3] = √[1411400/3] = √470467 = 686. z=(420−1390)/686 = −1.41. Small N makes z less reliable but the −77.6% deviation is unambiguous.

Step 4 — Weekend pattern: Sat consistently highest, Sun lowest — normal consumer e-commerce pattern.""",
    answer="""## E-Commerce Transaction Anomaly Detection

### Full Dataset

| Day | Wk 1 | Wk 2 | Wk 3 | 2-Wk Baseline | Wk3 Deviation |
|-----|------|------|------|---------------|---------------|
| Mon | 1,820 | 1,790 | 1,810 | 1,805 | +0.3% — Normal |
| Tue | 1,950 | 1,930 | 1,960 | 1,940 | +1.0% — Normal |
| **Wed** | **1,880** | **1,870** | **420** | **1,875** | **−77.6% 🔴 ANOMALY** |
| Thu | 1,910 | 1,890 | 1,920 | 1,900 | +1.1% — Normal |
| Fri | 2,150 | 2,180 | 2,200 | 2,165 | +1.6% — Normal |
| Sat | 2,400 | 2,350 | 2,410 | 2,375 | +1.5% — Normal |
| Sun | 1,600 | 1,580 | 1,590 | 1,590 | 0.0% — Normal |

### Assessment

**Week 3 Wednesday (420 transactions) is a critical anomaly.**

1. **Magnitude:** 420 vs. baseline 1,875 — a loss of ~1,455 transactions (77.6% drop).
2. **Isolation:** Thursday returned immediately to normal (1,920), ruling out a multi-day event. The anomaly is confined to exactly one day.
3. **Normal weekly pattern confirmed:** Sat > Fri > Thu ≈ Tue ≈ Mon ≈ Wed > Sun — consistent across all three weeks (excluding the anomaly).

### Likely Root Causes (in priority order)

| Hypothesis | Evidence | Probability |
|------------|----------|-------------|
| Platform outage / downtime | Transactions near zero for one day, immediate recovery | **High** |
| Payment gateway failure | Same profile as outage but partial | Medium |
| Data pipeline error (missing data) | 420 may be partial-day count | Medium |
| Genuine demand drop | No external event; Thu/Fri normal | Low |

### Recommended Actions

- Pull infrastructure logs for Wk3 Wednesday — check for deployment, DB, or CDN incidents.
- Cross-reference with revenue data — if revenue also dropped, it's real; if revenue is normal, it's a logging gap.
- Estimate revenue impact: ~1,455 lost transactions × average order value.

**Confidence: High** — anomaly is statistically unambiguous; root cause requires log investigation."""
))

examples.append(make(
    query="""Accounts receivable aging report:

Customer | Current | 30-day | 60-day | 90-day | 90+ day | Total
Acme Corp | 45000 | 12000 | 0 | 0 | 0 | 57000
Beta Ltd | 28000 | 8500 | 0 | 0 | 0 | 36500
Gamma Inc | 15000 | 22000 | 18000 | 12000 | 0 | 67000
Delta Co | 32000 | 0 | 0 | 0 | 45000 | 77000
Echo PLC | 19000 | 5000 | 2500 | 0 | 0 | 26500
Total | 139000 | 47500 | 20500 | 12000 | 45000 | 264000

Identify anomalies, calculate DSO for each customer, flag collection risks, and prioritise actions.""",
    reasoning="""DSO = (Total AR / Annual Revenue) × 365. Without revenue data, use bucket analysis as proxy.

Anomaly scan — look for unusual aging patterns:
  Acme: clean — only current and 30-day. Normal.
  Beta: clean — current and 30-day. Normal.
  Gamma: has 30+60+90-day buckets totalling 52k vs 15k current. 77.6% of balance is overdue. Escalating delinquency.
  Delta: 45k in 90+ day, zero in 30/60/90 — no intermediate aging. This means the 90+ debt is either very old (skipped buckets) or a disputed invoice. No new current → possibly no new orders.
  Echo: minor overdue (30+60 = 7500/26500 = 28.3%). Manageable.

Overdue % per customer:
  Acme: 12000/57000 = 21.1%
  Beta: 8500/36500 = 23.3%
  Gamma: 52000/67000 = 77.6%
  Delta: 45000/77000 = 58.4%
  Echo: 7500/26500 = 28.3%
  Portfolio: 125000/264000 = 47.3%

Delta anomaly: 90+ with nothing in 30/60/90 is unusual — suggests a lump-sum dispute, not rolling overdue.""",
    answer="""## Accounts Receivable Aging — Anomaly & Risk Analysis

### Aging Summary

| Customer | Current | 1–30d | 31–60d | 61–90d | 90+d | Total | % Overdue |
|----------|---------|-------|--------|--------|------|-------|-----------|
| Acme Corp | $45,000 | $12,000 | — | — | — | $57,000 | 21.1% |
| Beta Ltd | $28,000 | $8,500 | — | — | — | $36,500 | 23.3% |
| Gamma Inc | $15,000 | $22,000 | $18,000 | $12,000 | — | $67,000 | **77.6%** |
| Delta Co | $32,000 | — | — | — | $45,000 | $77,000 | **58.4%** |
| Echo PLC | $19,000 | $5,000 | $2,500 | — | — | $26,500 | 28.3% |
| **Total** | **$139,000** | **$47,500** | **$20,500** | **$12,000** | **$45,000** | **$264,000** | **47.3%** |

### Anomalies Identified

**1. Delta Co — 90+ day with empty intermediate buckets (CRITICAL)**
- $45,000 sits in 90+ with zero in the 30/60/90-day buckets. This is statistically impossible under normal rolling overdue — it indicates either:
  - A single old disputed invoice that has never moved, **or**
  - A data entry error where the invoice was mis-dated.
- Delta also has no 30/60/90-day balance — suggesting they may have halted normal trading or are paying current invoices but disputing the old one.

**2. Gamma Inc — Escalating delinquency cascade (HIGH RISK)**
- Balance distributed across all aging buckets in decreasing amounts ($22k → $18k → $12k), consistent with a customer progressively slowing payments. 77.6% overdue.

### Risk Prioritisation

| Priority | Customer | Exposure | Action |
|----------|----------|----------|--------|
| 🔴 1 | Delta Co | $45,000 (90+) | Contact immediately — confirm dispute status; consider legal notice |
| 🔴 2 | Gamma Inc | $52,000 overdue | Payment plan or credit hold; stop new shipments |
| 🟡 3 | Acme Corp | $12,000 (30d) | Routine follow-up |
| 🟡 4 | Echo PLC | $7,500 overdue | Standard reminder |
| 🟢 5 | Beta Ltd | $8,500 (30d) | Within normal terms |

**Total overdue: $125,000 (47.3% of AR portfolio)** — above healthy threshold of ~25–30%.

**Confidence: High** — aging analysis based on provided data; DSO calculation requires revenue data not provided."""
))

examples.append(make(
    query="""Sales rep performance data — units sold per month for 6 reps over 6 months:

       Jan  Feb  Mar  Apr  May  Jun
Alice  82   88   91   85   93   96
Bob    75   72   78   74   71   69
Carol  90   92   88   94   97   95
Dave   85   87   83   88   12   91
Eve    70   68   72   71   69   73
Frank  95   98   102  99   105  108

Detect anomalies, assess each rep's trajectory, and identify who needs attention.""",
    reasoning="""Step 1 — Scan for obvious outliers: Dave April=12 is far below his range (83–91). Likely data error or absence/illness.

Step 2 — Trend direction per rep:
  Alice: 82→96, slope ≈ +2.8/month, positive
  Bob: 75→69, slope ≈ −1.2/month, declining
  Carol: 90→95, with variability, slightly positive
  Dave: excluding the 12, 85→91, positive — but 12 in May is anomalous
  Eve: 70→73, mostly flat/slight positive (range 68–73)
  Frank: 95→108, slope ≈ +2.6/month, strong positive

Step 3 — Z-score for Dave May: Dave's values (excl May) = 85,87,83,88,91; mean=86.8, σ=2.86; z=(12−86.8)/2.86 = −26.2. Extreme outlier.

Step 4 — Bob trajectory: 6 consecutive months declining (75,72,78,74,71,69) — Feb-Mar uptick doesn't break the downward trend. Linear fit: roughly −1.2/month.

Step 5 — Ranking by Jun performance: Frank(108)>Carol(95)>Alice(96)... wait: Alice Jun=96, Carol Jun=95. Frank=108, Alice=96, Carol=95, Dave=91, Eve=73, Bob=69.""",
    answer="""## Sales Rep Performance Analysis — Anomaly & Trend Detection

### Monthly Units Sold

| Rep | Jan | Feb | Mar | Apr | May | Jun | Avg | Trend |
|-----|-----|-----|-----|-----|-----|-----|-----|-------|
| Alice | 82 | 88 | 91 | 85 | 93 | 96 | 89.2 | ↑ +2.8/mo |
| Bob | 75 | 72 | 78 | 74 | 71 | 69 | 73.2 | ↓ −1.2/mo |
| Carol | 90 | 92 | 88 | 94 | 97 | 95 | 92.7 | ↑ Stable-high |
| **Dave** | **85** | **87** | **83** | **88** | **12** | **91** | **74.3*** | **⚠ Anomaly** |
| Eve | 70 | 68 | 72 | 71 | 69 | 73 | 70.5 | → Flat |
| Frank | 95 | 98 | 102 | 99 | 105 | 108 | 101.2 | ↑ +2.6/mo |

*Dave's average is severely distorted by the May anomaly; excluding May: avg = 86.8.

### Anomaly: Dave — May = 12 Units (z = −26.2)

Dave's normal range is 83–91 units/month. May's value of **12** is 74.8 units below his mean — a z-score of −26.2. This is not a performance issue; it is **almost certainly a data error** (keying error, missing data, or Dave was absent for most of May). His Jun recovery to 91 confirms no underlying performance problem.

**Action:** Verify May data with source system. If absence: note and exclude from performance calculations.

### Performance Rankings (Jun)

| Rank | Rep | Jun Units | 6-Month Trend | Status |
|------|-----|-----------|---------------|--------|
| 1 | Frank | 108 | Strong growth | ⭐ Top performer |
| 2 | Alice | 96 | Steady growth | ✓ On track |
| 3 | Carol | 95 | Consistent high | ✓ Reliable |
| 4 | Dave | 91 | Growing (excl May) | ⚠ Data anomaly to resolve |
| 5 | Eve | 73 | Flat | 🟡 Coaching opportunity |
| 6 | Bob | 69 | Declining | 🔴 Needs intervention |

### Attention Required

- **Bob:** Six-month decline is sustained and not reversing. Performance coaching or territory review needed.
- **Eve:** Flat performance — not declining, but no growth. Motivation or skill-gap assessment warranted.
- **Dave May data:** Must be corrected before any incentive calculations.

**Confidence: High** — all trends and anomalies identified from provided data; Dave May conclusion requires data verification."""
))

# ── PIVOT / SUMMARY FROM RAW DATA (4) ────────────────────────────────────────

examples.append(make(
    query="""Raw transaction log (15 rows):

TxID | Date     | Region | Product    | Qty | UnitPrice
T001 | 2024-01-05 | North | Widget-A | 10 | 25
T002 | 2024-01-08 | South | Widget-B | 5  | 40
T003 | 2024-01-12 | North | Widget-A | 8  | 25
T004 | 2024-01-15 | West  | Widget-C | 20 | 15
T005 | 2024-01-18 | South | Widget-A | 12 | 25
T006 | 2024-01-22 | North | Widget-B | 6  | 40
T007 | 2024-01-25 | West  | Widget-A | 15 | 25
T008 | 2024-02-02 | South | Widget-C | 10 | 15
T009 | 2024-02-07 | North | Widget-A | 20 | 25
T010 | 2024-02-11 | West  | Widget-B | 8  | 40
T011 | 2024-02-14 | South | Widget-A | 7  | 25
T012 | 2024-02-19 | North | Widget-C | 25 | 15
T013 | 2024-02-22 | West  | Widget-A | 11 | 25
T014 | 2024-02-26 | South | Widget-B | 4  | 40
T015 | 2024-02-28 | North | Widget-A | 9  | 25

Build a pivot table: Revenue by Region × Product, plus totals. Then pivot by Month × Region.""",
    reasoning="""Revenue = Qty × UnitPrice per transaction.

Compute per transaction:
  T001: N, WA, 10×25=250    T002: S, WB, 5×40=200     T003: N, WA, 8×25=200
  T004: W, WC, 20×15=300    T005: S, WA, 12×25=300    T006: N, WB, 6×40=240
  T007: W, WA, 15×25=375    T008: S, WC, 10×15=150    T009: N, WA, 20×25=500
  T010: W, WB, 8×40=320     T011: S, WA, 7×25=175     T012: N, WC, 25×15=375
  T013: W, WA, 11×25=275    T014: S, WB, 4×40=160     T015: N, WA, 9×25=225

Pivot Region × Product:
  North–WA: T001+T003+T009+T015 = 250+200+500+225 = 1175
  North–WB: T006 = 240
  North–WC: T012 = 375
  North total: 1790
  South–WA: T005+T011 = 300+175 = 475
  South–WB: T002+T014 = 200+160 = 360
  South–WC: T008 = 150
  South total: 985
  West–WA: T007+T013 = 375+275 = 650
  West–WB: T010 = 320
  West–WC: T004 = 300
  West total: 1270
  Grand total: 1790+985+1270 = 4045

Month × Region:
  Jan: North(T001+T003+T006)=250+200+240=690, South(T002+T005)=200+300=500, West(T004+T007)=300+375=675; Jan total=1865
  Feb: North(T009+T012+T015)=500+375+225=1100, South(T008+T011+T014)=150+175+160=485, West(T010+T013)=320+275=595; Feb total=2180""",
    answer="""## Pivot Table Analysis — Revenue by Region & Product

### Pivot 1: Revenue by Region × Product

| Region | Widget-A | Widget-B | Widget-C | **Region Total** |
|--------|----------|----------|----------|-----------------|
| North | $1,175 | $240 | $375 | **$1,790** |
| South | $475 | $360 | $150 | **$985** |
| West | $650 | $320 | $300 | **$1,270** |
| **Product Total** | **$2,300** | **$920** | **$825** | **$4,045** |

### Pivot 2: Revenue by Month × Region

| Month | North | South | West | **Month Total** |
|-------|-------|-------|------|----------------|
| January | $690 | $500 | $675 | **$1,865** |
| February | $1,100 | $485 | $595 | **$2,180** |
| **Region Total** | **$1,790** | **$985** | **$1,270** | **$4,045** |

### Key Insights

1. **Widget-A dominates revenue** — $2,300 of $4,045 (56.9%). It's the highest-volume and most widespread product.
2. **North is the strongest region** ($1,790, 44.2%), driven almost entirely by Widget-A ($1,175).
3. **February grew 16.9% over January** ($2,180 vs $1,865) — North drove this with a +59.4% MoM jump.
4. **South February dipped slightly** ($485 vs $500) despite company-wide growth — warrants attention.
5. **Widget-C is underperforming** in South ($150 total, only 1 transaction) — limited penetration.

**Confidence: High** — all 15 transactions verified; pivot totals cross-check to $4,045."""
))

examples.append(make(
    query="""HR raw data — 20 employee records:
Dept, Level, Gender, Salary
Eng,Senior,M,120000 | Eng,Senior,F,115000 | Eng,Mid,M,90000 | Eng,Mid,F,88000 | Eng,Junior,M,65000
Eng,Junior,F,63000  | Sales,Senior,M,105000 | Sales,Senior,F,98000 | Sales,Mid,M,78000 | Sales,Mid,F,76000
Sales,Junior,M,55000 | Sales,Junior,F,54000 | HR,Senior,M,95000 | HR,Senior,F,92000 | HR,Mid,M,72000
HR,Mid,F,70000 | HR,Junior,M,50000 | HR,Junior,F,49000 | Ops,Senior,M,88000 | Ops,Senior,F,85000

Build a pivot of avg salary by Dept × Level, calculate gender pay gap at each level across all depts, and flag concerns.""",
    reasoning="""Pivot Dept × Level (average salary):
  Eng Senior: M=120k, F=115k; avg=(120+115)/2=117.5k
  Eng Mid: M=90k, F=88k; avg=89k
  Eng Junior: M=65k, F=63k; avg=64k
  Sales Senior: M=105k, F=98k; avg=101.5k
  Sales Mid: M=78k, F=76k; avg=77k
  Sales Junior: M=55k, F=54k; avg=54.5k
  HR Senior: M=95k, F=92k; avg=93.5k
  HR Mid: M=72k, F=70k; avg=71k
  HR Junior: M=50k, F=49k; avg=49.5k
  Ops Senior: M=88k, F=85k; avg=86.5k

Gender pay gap (F as % of M salary, by level across all depts):
  Senior: (115+98+92+85)/(120+105+95+88) = 390/408 = 95.6% → gap = 4.4%
  Mid: (88+76+70)/(90+78+72) = 234/240 = 97.5% → gap = 2.5%
  Junior: (63+54+49)/(65+55+50) = 166/170 = 97.6% → gap = 2.4%

Overall gap is largest at Senior level (4.4%). Sales Senior gap: (98−105)/105=−6.7% — widest single pair.""",
    answer="""## HR Salary Pivot — Dept × Level with Gender Pay Gap Analysis

### Pivot 1: Average Salary by Dept × Level

| Dept | Junior | Mid | Senior |
|------|--------|-----|--------|
| Engineering | $64,000 | $89,000 | $117,500 |
| Sales | $54,500 | $77,000 | $101,500 |
| HR | $49,500 | $71,000 | $93,500 |
| Operations | — | — | $86,500 |

### Pivot 2: Salary by Gender × Level

| Level | Male Avg | Female Avg | F as % of M | **Pay Gap** |
|-------|----------|------------|-------------|-------------|
| Senior | $102,000 | $97,500 | 95.6% | **−4.4%** |
| Mid | $80,000 | $78,000 | 97.5% | −2.5% |
| Junior | $56,667 | $55,333 | 97.6% | −2.4% |

### Dept-Level Pay Gap Detail

| Dept | Level | M Salary | F Salary | Gap |
|------|-------|----------|----------|-----|
| Engineering | Senior | $120,000 | $115,000 | −4.2% |
| **Sales** | **Senior** | **$105,000** | **$98,000** | **−6.7%** |
| HR | Senior | $95,000 | $92,000 | −3.2% |
| Operations | Senior | $88,000 | $85,000 | −3.4% |

### Findings & Concerns

1. **Gender pay gap exists at every level and in every department** — no exceptions. All females earn less than equivalent-level males.
2. **Gap widens at Senior level (4.4%)** vs. Mid/Junior (~2.4–2.5%), indicating a promotion/negotiation differential compounding over careers.
3. **Sales Senior is the most concerning** at −6.7% ($7,000 gap) — the largest single pair in the dataset.
4. **The pattern is systematic**, not random — consistent direction across 10 comparable pairs (out of 10). Statistical probability of this occurring by chance: (0.5)^10 = 0.1%.

### Recommended Actions

- Conduct formal pay equity audit with significance testing on full employee population.
- Review Sales Senior compensation band — the $7,000 gap may violate pay equity legislation in some jurisdictions.
- Implement structured salary bands with transparent criteria to eliminate negotiation-driven gaps.

**Confidence: High** — all calculations from provided data; caveats: small sample size (N=20), no tenure/performance controls applied."""
))

examples.append(make(
    query="""Raw support ticket data (18 tickets):
TicketID, Category, Priority, AgentID, ResolutionTime_hrs
T1,Billing,High,A1,4  | T2,Technical,Critical,A2,2  | T3,Billing,Low,A1,24
T4,Technical,High,A3,6 | T5,Account,Medium,A2,8 | T6,Technical,Critical,A1,3
T7,Billing,Medium,A3,12 | T8,Account,High,A2,5 | T9,Technical,Low,A1,18
T10,Billing,Critical,A3,1 | T11,Technical,Medium,A2,10 | T12,Account,Low,A3,30
T13,Technical,High,A1,7 | T14,Billing,Medium,A2,9 | T15,Account,Critical,A3,2
T16,Technical,Low,A1,22 | T17,Billing,High,A2,5 | T18,Account,Medium,A3,11

Build pivot: avg resolution time by Category × Priority. Also avg by Agent. Identify SLA risks.""",
    reasoning="""Step 1 — Group by Category × Priority and compute avg resolution time:
  Billing-High: T1(4), T17(5) → avg=4.5h
  Billing-Low: T3(24) → avg=24h
  Billing-Medium: T7(12), T14(9) → avg=10.5h
  Billing-Critical: T10(1) → avg=1h
  Technical-Critical: T2(2), T6(3) → avg=2.5h
  Technical-High: T4(6), T13(7) → avg=6.5h
  Technical-Low: T9(18), T16(22) → avg=20h
  Technical-Medium: T11(10) → avg=10h
  Account-Critical: T15(2) → avg=2h
  Account-High: T8(5) → avg=5h
  Account-Low: T12(30) → avg=30h
  Account-Medium: T5(8), T18(11) → avg=9.5h

Step 2 — By agent:
  A1: T1(4),T3(24),T6(3),T9(18),T13(7),T16(22) → sum=78, n=6, avg=13.0h
  A2: T2(2),T5(8),T8(5),T11(10),T14(9),T17(5) → sum=39, n=6, avg=6.5h
  A3: T4(6),T7(12),T10(1),T12(30),T15(2),T18(11) → sum=62, n=6, avg=10.3h

Step 3 — Typical SLA targets: Critical≤4h, High≤8h, Medium≤24h, Low≤48h
  SLA breaches: Billing-Low=24h (borderline 24h SLA), Technical-Low=20h, Account-Low=30h — all Low priority but still check.
  No Critical or High breaches observed.""",
    answer="""## Support Ticket Pivot Analysis

### Pivot 1: Avg Resolution Time (hrs) by Category × Priority

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Billing | 1.0h | 4.5h | 10.5h | **24.0h** |
| Technical | 2.5h | 6.5h | 10.0h | **20.0h** |
| Account | 2.0h | 5.0h | 9.5h | **30.0h** |

### Pivot 2: Avg Resolution Time by Agent

| Agent | Tickets | Avg Resolution | Performance |
|-------|---------|---------------|-------------|
| A2 | 6 | **6.5h** | Best |
| A3 | 6 | 10.3h | Mid |
| A1 | 6 | **13.0h** | Needs review |

### SLA Analysis (standard targets: Critical 4h / High 8h / Medium 24h / Low 48h)

| Priority | Target | Actual Avg | Status |
|----------|--------|-----------|--------|
| Critical | ≤4h | 1.8h avg | ✅ Well within SLA |
| High | ≤8h | 5.3h avg | ✅ Within SLA |
| Medium | ≤24h | 10h avg | ✅ Within SLA |
| Low | ≤48h | 24.7h avg | ✅ Within SLA (marginally) |

### Key Findings

1. **No SLA breaches on Critical or High priority** — the team correctly prioritises urgent tickets.
2. **Low-priority tickets are slow across all categories** (20–30h), with Account-Low at 30h being the slowest. While within the 48h SLA, Account-Low's single data point (T12=30h) warrants monitoring.
3. **Agent A1 averages 13.0h vs A2's 6.5h** — a 2× performance gap. A1 handles two 22h+ low-priority tickets (T3, T16) which inflate the average; however even high-priority tickets take longer for A1.
4. **Billing-Critical resolved in 1h** — outstanding performance, possibly due to automated billing workflows.

### Recommendations

- Review A1's ticket queue and workload; shadow A2 for process benchmarking.
- Set agent-level SLA dashboards so performance differences surface in real time.
- Add more data points for Account-Low before drawing firm conclusions (N=1).

**Confidence: High** — all pivot calculations verified against the 18 source records."""
))

examples.append(make(
    query="""Raw sales pipeline data (16 opportunities):
OppID | Stage | Value | Probability | Owner | CloseMonth
O1|Qualified|50000|0.20|Alice|Mar | O2|Proposal|120000|0.50|Bob|Feb | O3|Negotiation|80000|0.75|Alice|Feb
O4|Closed-Won|95000|1.00|Carol|Jan | O5|Qualified|30000|0.20|Bob|Apr | O6|Proposal|60000|0.50|Carol|Mar
O7|Negotiation|110000|0.75|Alice|Feb | O8|Qualified|45000|0.20|Carol|Apr | O9|Proposal|75000|0.50|Bob|Mar
O10|Closed-Won|140000|1.00|Alice|Jan | O11|Negotiation|55000|0.75|Carol|Mar | O12|Proposal|90000|0.50|Alice|Apr
O13|Closed-Won|70000|1.00|Bob|Jan | O14|Qualified|25000|0.20|Carol|May | O15|Negotiation|65000|0.75|Bob|Apr | O16|Proposal|85000|0.50|Carol|Feb

Build pivot by Stage (count, total value, weighted pipeline). Also pivot by Owner (weighted pipeline, closed revenue). Identify top performer and pipeline health.""",
    reasoning="""Weighted value = Value × Probability per opportunity.

By Stage:
  Qualified (0.20): O1(50k), O5(30k), O8(45k), O14(25k) → count=4, total=150k, weighted=30k
  Proposal (0.50): O2(120k), O6(60k), O9(75k), O12(90k), O16(85k) → count=5, total=430k, weighted=215k
  Negotiation (0.75): O3(80k), O7(110k), O11(55k), O15(65k) → count=4, total=310k, weighted=232.5k
  Closed-Won (1.00): O4(95k), O10(140k), O13(70k) → count=3, total=305k, weighted=305k
  Grand: 16 opps, total pipeline=1195k, weighted=782.5k

By Owner — weighted pipeline (all stages incl closed):
  Alice: O1(50k×0.2=10k), O3(80k×0.75=60k), O7(110k×0.75=82.5k), O10(140k×1.0=140k), O12(90k×0.5=45k)
    Total weighted = 10+60+82.5+140+45 = 337.5k; Closed = 140k (O10 only)
  Bob: O2(120k×0.5=60k), O5(30k×0.2=6k), O9(75k×0.5=37.5k), O13(70k×1.0=70k), O15(65k×0.75=48.75k)
    Total weighted = 60+6+37.5+70+48.75 = 222.25k; Closed = 70k
  Carol: O4(95k×1.0=95k), O6(60k×0.5=30k), O8(45k×0.2=9k), O11(55k×0.75=41.25k), O14(25k×0.2=5k), O16(85k×0.5=42.5k)
    Total weighted = 95+30+9+41.25+5+42.5 = 222.75k; Closed = 95k""",
    answer="""## Sales Pipeline Pivot Analysis

### Pivot 1: Pipeline by Stage

| Stage | Count | Total Value | Probability | **Weighted Pipeline** |
|-------|-------|-------------|-------------|----------------------|
| Qualified | 4 | $150,000 | 20% | $30,000 |
| Proposal | 5 | $430,000 | 50% | $215,000 |
| Negotiation | 4 | $310,000 | 75% | $232,500 |
| Closed-Won | 3 | $305,000 | 100% | $305,000 |
| **Total** | **16** | **$1,195,000** | — | **$782,500** |

### Pivot 2: Pipeline by Owner

| Owner | Opps | Total Value | Weighted Pipeline | Closed Revenue | Win Share |
|-------|------|-------------|-------------------|---------------|-----------|
| **Alice** | 5 | $470,000 | **$337,500** | **$140,000** | 45.9% |
| Carol | 6 | $370,000 | $222,750 | $95,000 | 31.1% |
| Bob | 5 | $355,000 | $222,250 | $70,000 | 23.0% |
| **Total** | **16** | **$1,195,000** | **$782,500** | **$305,000** | 100% |

### Pipeline Health Assessment

**1. Top performer: Alice**
- Highest weighted pipeline ($337,500 — 43.1% of total).
- Highest closed revenue ($140,000 — O10 alone).
- Has two high-value Negotiation deals (O3: $80k, O7: $110k) closing in February.

**2. Near-term revenue (Feb close month):**
  O2 (Bob, $120k, 50%), O3 (Alice, $80k, 75%), O7 (Alice, $110k, 75%), O16 (Carol, $85k, 50%)
  Weighted Feb: 60k + 60k + 82.5k + 42.5k = **$245,000 expected in February**

**3. Pipeline funnel health:**
- Proposal stage is the largest by count (5) and has healthy conversion to Negotiation.
- Qualified stage (4 opps, $30k weighted) is thin — BD activity needed to fill top of funnel.
- Negotiation-to-Close conversion rate (historical): 3 Closed-Won vs 4 in Negotiation — implies ~75% close rate, consistent with the probability assigned.

**4. Carol has the most opps (6) but lowest weighted pipeline per opp** ($37,125 vs Alice's $67,500) — her deals skew smaller.

**Confidence: High** — all weighted calculations verified; probability assumptions are those provided in source data."""
))

with open(OUT, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Written {len(examples)} examples to {OUT}")
