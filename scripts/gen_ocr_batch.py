"""
Generate 20 OCR/degraded document training examples for DocWain.
Categories (4 each):
  1. scanned_receipt   — 0→O, 1→l substitutions
  2. faded_invoice     — missing chars / partial words
  3. handwritten_mixed — handwritten + printed hybrid
  4. lowres_table      — low-res scanned tables with noise
  5. stamped_official  — official docs with heavy stamp overlay noise
"""

import json, pathlib, textwrap

OUT = pathlib.Path(
    "/home/ubuntu/PycharmProjects/DocWain/finetune_artifacts/sprint/claude_quality/ocr_vision_batch.jsonl"
)

SYS = (
    "You are DocWain, an enterprise document intelligence assistant. "
    "You analyse documents with deep contextual understanding, extract structured information, "
    "identify patterns and anomalies, and provide holistic analysis grounded in evidence. "
    "You reason step-by-step before answering, state your confidence level, and cite specific sources. "
    "When information is insufficient, you say so clearly rather than guessing."
)

def fmt(query: str, reasoning: str, answer: str) -> dict:
    text = (
        f"<|im_start|>system\n{SYS}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{textwrap.dedent(reasoning).strip()}\n</think>\n\n"
        f"{textwrap.dedent(answer).strip()}<|im_end|>"
    )
    return {"text": text, "category": "ocr_vision", "difficulty": "hard", "source": "claude_quality"}


# ── 1. SCANNED RECEIPTS (0→O, 1→l) ──────────────────────────────────────────

examples = []

examples.append(fmt(
    query=(
        "Extract total amount, date, and merchant name from this scanned receipt:\n\n"
        "WHOLE F00DS MARKET\n"
        "0ak Street, Ch1cago IL 6061O\n"
        "Date: 04/1O/2O26  Time: 14:3l\n"
        "Organic M1lk      $4.99\n"
        "Wh0le Grain Bread  $3.49\n"
        "Fr0zen P1zza       $7.99\n"
        "Sub-T0tal:        $16.47\n"
        "Tax (lO%):         $1.65\n"
        "T0TAL:            $18.12\n"
        "Thank y0u f0r sh0pping!"
    ),
    reasoning="""
        The receipt has systematic OCR character substitution errors typical of degraded thermal paper scans.
        I see '0' used in place of 'O' and '1' used in place of 'l' throughout.
        Specific errors: 'F00DS'→'FOODS', '0ak'→'Oak', 'Ch1cago'→'Chicago', '6061O'→'60610',
        '04/1O/2O26'→'04/10/2026', '14:3l'→'14:31', 'M1lk'→'Milk', 'Wh0le'→'Whole',
        'Fr0zen'→'Frozen', 'P1zza'→'Pizza', 'Sub-T0tal'→'Sub-Total', 'lO%'→'10%',
        'T0TAL'→'TOTAL', 'y0u'→'you', 'f0r'→'for', 'sh0pping'→'shopping'.
        Math check: $4.99+$3.49+$7.99=$16.47 ✓; 10% of $16.47=$1.647≈$1.65 ✓; total $18.12 ✓.
        The zip code '60610' is a real Chicago ZIP — confirms the address correction.
        Merchant name is clearly Whole Foods Market after correction.
    """,
    answer="""
        **Extracted Fields (OCR-corrected):**

        | Field           | Raw OCR Value         | Corrected Value          | Confidence |
        |-----------------|----------------------|--------------------------|------------|
        | Merchant Name   | WHOLE F00DS MARKET   | WHOLE FOODS MARKET       | High       |
        | Address         | 0ak Street Ch1cago IL 6061O | Oak Street, Chicago IL 60610 | High |
        | Date            | 04/1O/2O26           | 04/10/2026               | High       |
        | Time            | 14:3l                | 14:31                    | High       |
        | Sub-Total       | $16.47               | $16.47                   | High       |
        | Tax             | lO% = $1.65          | 10% = $1.65              | High       |
        | **Total**       | $18.12               | **$18.12**               | High       |

        **OCR Error Pattern:** Digit '0' substituted for letter 'O' (15 instances); digit '1' substituted for letter 'l' (4 instances). Classic thermal-paper scanner degradation.

        **Arithmetic Validation:** $4.99 + $3.49 + $7.99 = $16.47 ✓ | Tax 10% = $1.647 ≈ $1.65 ✓ | Total $18.12 ✓

        **Confidence:** High overall. All amounts cross-validate. Address ZIP 60610 confirmed as Chicago, IL.
    """
))

examples.append(fmt(
    query=(
        "Parse this scanned petrol station receipt and extract vehicle info and fuel cost:\n\n"
        "SHELl PETROl STATI0N #4821\n"
        "Highw4y 1O1, San Fr4nc1sc0, CA\n"
        "Pump: O3    Grade: Prem1um\n"
        "Reg. Plate: 7AB1234\n"
        "Gall0ns: 12.Ol  @ $4.l9/gal\n"
        "Am0unt Due: $5O.37\n"
        "Card: VISA ...l234\n"
        "Auth: 0K9921"
    ),
    reasoning="""
        This petrol station receipt has both 0/O and 1/l substitutions plus a digit/letter swap in the highway number.
        'SHELl'→'SHELL', 'PETROl'→'PETROL', 'STATI0N'→'STATION'.
        '1O1'→'101' (Highway 101 in California — well-known route, confirms correction).
        'Fr4nc1sc0'→'Francisco' — the '4' is an OCR artifact for 'a', '1' for 'i', '0' for 'o'.
        'Pump: O3'→'03', 'Prem1um'→'Premium', 'Gall0ns'→'Gallons', '12.Ol'→'12.01'.
        '$4.l9'→'$4.19' (the 'l' is a '1').
        Math check: 12.01 × $4.19 = $50.32 — slight rounding: $50.37 plausible with mid-grade pricing.
        'Am0unt'→'Amount', '$5O.37'→'$50.37', 'l234'→'1234', 'O K9921'→'OK9921'.
        Plate '7AB1234' appears clean — no substitutions evident.
    """,
    answer="""
        **Extracted Fields (OCR-corrected):**

        | Field          | Corrected Value          | Confidence |
        |----------------|--------------------------|------------|
        | Merchant       | SHELL PETROL STATION #4821 | High     |
        | Location       | Highway 101, San Francisco, CA | High |
        | Pump Number    | 03                       | High       |
        | Fuel Grade     | Premium                  | High       |
        | Vehicle Plate  | 7AB1234                  | High       |
        | Volume         | 12.01 gallons            | High       |
        | Unit Price     | $4.19/gallon             | High       |
        | **Total**      | **$50.37**               | High       |
        | Payment        | VISA ending 1234         | High       |
        | Auth Code      | OK9921                   | Medium     |

        **OCR Errors Detected:** 0↔O (8 instances), 1↔l (6 instances), digit '4' for letter 'a' (2 instances in 'Francisco').

        **Note:** 12.01 × $4.19 = $50.32; $50.37 difference of $0.05 may reflect partial gallon rounding or per-transaction fee. Flag for review if exact reconciliation needed.

        **Confidence:** High. Highway 101 and San Francisco are geographically consistent.
    """
))

examples.append(fmt(
    query=(
        "Extract line items and totals from this grocery store receipt with OCR noise:\n\n"
        "COST C0 WHOLES4LE\n"
        "Mem #: 1l23456789\n"
        "lt: Kirkl4nd 0live 0il 2L   $l4.99\n"
        "lt: Wh0lesale B0x Water(24) $6.99\n"
        "lt: 0rg4nic Eggs Dz          $8.49\n"
        "lt: Cheddar Cheese Blk 2lb  $1O.99\n"
        "lt: Gr0und C0ffee 28oz      $13.49\n"
        "SubT0tal                    $54.95\n"
        "Tax                          $O.OO\n"
        "T0tal                       $54.95\n"
        "Cash Tend                   $6O.OO\n"
        "Change                       $5.O5"
    ),
    reasoning="""
        Costco receipt with heavy 0/O and 1/l substitution, plus digit '4' for 'a'.
        'C0ST C0'→'COST CO' (Costco), 'WHOLES4LE'→'WHOLESALE', 'Mem #: 1l23456789'→'1123456789'.
        'lt:'→'lt:' — likely 'lt' is 'Item' abbreviated; consistent formatting across all lines, keep as-is but note it.
        'Kirkl4nd'→'Kirkland' (Costco's house brand — confirms retailer identity).
        '0live 0il'→'Olive Oil', '$l4.99'→'$14.99', 'Wh0lesale'→'Wholesale', 'B0x'→'Box'.
        '0rg4nic'→'Organic', '0.OO'→'0.00' (tax exempt — Costco food items typically tax-exempt in many US states).
        'T0tal'→'Total', '$6O.OO'→'$60.00', '$5.O5'→'$5.05'.
        Math: $14.99+$6.99+$8.49+$10.99+$13.49 = $54.95 ✓; $60.00-$54.95=$5.05 ✓.
    """,
    answer="""
        **Merchant:** COST CO WHOLESALE (Costco) | **Member #:** 1123456789

        **Line Items (OCR-corrected):**

        | Item                        | Price   | Confidence |
        |-----------------------------|---------|------------|
        | Kirkland Olive Oil 2L       | $14.99  | High       |
        | Wholesale Box Water (24pk)  | $6.99   | High       |
        | Organic Eggs Dozen          | $8.49   | High       |
        | Cheddar Cheese Block 2lb    | $10.99  | High       |
        | Ground Coffee 28oz          | $13.49  | High       |

        **Totals:**
        - Sub-Total: $54.95 | Tax: $0.00 (food tax-exempt) | **Total: $54.95**
        - Cash Tendered: $60.00 | Change: $5.05

        **Arithmetic Validation:** Sum of items = $54.95 ✓ | Change = $60.00 − $54.95 = $5.05 ✓

        **OCR Errors:** 0↔O (12 instances), 1↔l (3 instances), 4↔a (3 instances in 'Kirkland', 'Wholesale', 'Organic').

        **Confidence:** High. All amounts mathematically consistent. Kirkland brand confirms Costco.
    """
))

examples.append(fmt(
    query=(
        "Extract merchant, items, and payment method from this degraded fast-food receipt:\n\n"
        "McD0NALDS #l2897\n"
        "0perat0r: T0m\n"
        "0rder #: 1O4l\n"
        "1x Big Mac C0mb0         $1O.49\n"
        "1x McFl0rry Van1lla       $3.99\n"
        "2x Apple P1e              $2.OO\n"
        "1x Large C0ke             $2.49\n"
        "lt Tax (8.25%)            $1.57\n"
        "T0TAL:                   $2O.54\n"
        "V1SA DEBIT: ...89l5\n"
        "APPR0VED\n"
        "Thank y0u, c0me aga1n!"
    ),
    reasoning="""
        McDonald's receipt with 0/O and 1/l substitutions throughout.
        'McD0NALDS'→'McDONALDS', '#l2897'→'#12897', '0perat0r'→'Operator', 'T0m'→'Tom'.
        '0rder #: 1O4l'→'Order #: 1041'.
        '1x Big Mac C0mb0'→'1x Big Mac Combo', '$1O.49'→'$10.49'.
        'McFl0rry'→'McFlurry', 'Van1lla'→'Vanilla'.
        '2x Apple P1e'→'2x Apple Pie', '$2.OO'→'$2.00'.
        'Large C0ke'→'Large Coke'.
        'lt Tax'→ likely 'Item Tax' or just 'Tax'; '8.25%' appears clean.
        Math: $10.49+$3.99+$2.00+$2.49 = $18.97; 8.25% tax on $18.97 = $1.565 ≈ $1.57 ✓; total $20.54 ✓.
        'V1SA'→'VISA', '...89l5'→'...8915', 'APPR0VED'→'APPROVED'.
        'y0u'→'you', 'c0me'→'come', 'aga1n'→'again'.
    """,
    answer="""
        **Extracted Fields (OCR-corrected):**

        | Field          | Corrected Value       | Confidence |
        |----------------|-----------------------|------------|
        | Merchant       | McDONALDS #12897      | High       |
        | Operator       | Tom                   | High       |
        | Order Number   | 1041                  | High       |

        **Line Items:**

        | Qty | Item              | Price  |
        |-----|-------------------|--------|
        | 1x  | Big Mac Combo     | $10.49 |
        | 1x  | McFlurry Vanilla  | $3.99  |
        | 2x  | Apple Pie         | $2.00  |
        | 1x  | Large Coke        | $2.49  |

        **Totals:** Sub-total $18.97 | Tax 8.25% = $1.57 | **Total: $20.54**

        **Payment:** VISA DEBIT ending 8915 — APPROVED

        **Arithmetic Check:** $10.49 + $3.99 + $2.00 + $2.49 = $18.97; 8.25% × $18.97 = $1.565 → $1.57 ✓; $18.97 + $1.57 = $20.54 ✓

        **OCR Errors:** 0↔O (14 instances), 1↔l (8 instances). Pattern consistent with worn dot-matrix printer head.

        **Confidence:** High. All figures mathematically validated.
    """
))

# ── 2. FADED INVOICES (missing characters) ────────────────────────────────────

examples.append(fmt(
    query=(
        "Extract invoice metadata and line items from this faded scanned invoice:\n\n"
        "INV IC  #: INV-2 26-04 12\n"
        "Da e: Apr  , 2026\n"
        "Due Da e: Apr 2 , 2026\n"
        "Vendor: Apex Sof wa e Sol  ions Ltd\n"
        "Bill o: Nor h S ar Cor ora ion\n"
        "Desc ip ion            Q y  Uni Price  To al\n"
        "En er rise Licens      5   $1,20 .00  $6,0 0.00\n"
        "Sup or  Con rac        1     $80 .00    $800.00\n"
        "Trai ing (2 days)      2     $4 0.00    $900.00\n"
        "Sub o al                              $7,7 0.00\n"
        "Tax (1 %)                               $770.00\n"
        "To al Due                             $8,4 0.00"
    ),
    reasoning="""
        This invoice has ink dropout/fading causing random missing characters — classic flatbed scanner with degraded toner.
        Reconstructing each field using context and arithmetic:
        'INV IC  #: INV-2 26-04 12'→'INVOICE #: INV-2026-0412' (date-formatted invoice number).
        'Da e: Apr  , 2026'→'Date: April [8-15], 2026' — day digit missing; 'Due Da e: Apr 2 , 2026'→'Due Date: Apr 2[5-9], 2026'.
        'Apex Sof wa e Sol  ions'→'Apex Software Solutions', 'Nor h S ar Cor ora ion'→'North Star Corporation'.
        'En er rise Licens  5  $1,20 .00  $6,0 0.00'→'Enterprise License x5 @ $1,200.00 = $6,000.00'.
        'Sup or  Con rac  1  $80 .00  $800.00'→'Support Contract x1 @ $800.00 = $800.00'.
        'Trai ing (2 days) 2  $4 0.00  $900.00'→'Training (2 days) x2 @ $450.00 = $900.00'.
        Arithmetic: $6,000 + $800 + $900 = $7,700; tax 10% = $770; total = $8,470.
        'Sub o al $7,7 0.00'→$7,700.00 ✓; 'Tax (1 %)' → '10%' (digit '0' dropped); 'To al Due $8,4 0.00'→$8,470.00 ✓.
    """,
    answer="""
        **Invoice Metadata (reconstructed from faded scan):**

        | Field         | Reconstructed Value                  | Confidence |
        |---------------|--------------------------------------|------------|
        | Invoice #     | INV-2026-0412                        | High       |
        | Invoice Date  | April [day illegible], 2026          | Medium     |
        | Due Date      | April 2[x], 2026 — day partially missing | Medium |
        | Vendor        | Apex Software Solutions Ltd          | High       |
        | Bill To       | North Star Corporation               | High       |

        **Line Items (reconstructed):**

        | Description               | Qty | Unit Price  | Total      | Confidence |
        |---------------------------|-----|-------------|------------|------------|
        | Enterprise License        | 5   | $1,200.00   | $6,000.00  | High       |
        | Support Contract          | 1   | $800.00     | $800.00    | High       |
        | Training (2 days)         | 2   | $450.00     | $900.00    | High       |

        **Totals:**
        - Sub-total: $7,700.00 | Tax (10%): $770.00 | **Total Due: $8,470.00**

        **Fading Pattern:** Consonant clusters 't', 'r', 'f', 'p' frequently dropped — consistent with low-toner flatbed scan. Arithmetic fully validates reconstructed amounts.

        **Confidence:** High on amounts (math-verified). Medium on exact invoice date (day digit lost).
    """
))

examples.append(fmt(
    query=(
        "Parse this faded utility invoice for account details and charges:\n\n"
        "ME RO  LECT IC  OWER  O.\n"
        "Acco n  N m er: 88 -442-  01\n"
        "Se vice  eriod: 01 Mar  026 –  1 Mar 2026\n"
        "Se vice  ddress: 47  Oak  idge  rive, D llas TX\n"
        "Elect icit  Usage:  450 k h\n"
        " ate:  0.112  er kWh\n"
        "Ene gy Cha ge:        $1 0.40\n"
        "Dist ibution Fee:      $22.5 \n"
        "Fue  Adjust ent:       $1 .20\n"
        "Tax s & Fees:          $1 .86\n"
        "Tota  Due:            $19 .96\n"
        "Due  ate: 30 Apr  026"
    ),
    reasoning="""
        Utility invoice with heavy character dropout from faded ink — multiple consonants and vowels missing.
        'ME RO  LECT IC  OWER  O.'→'METRO ELECTRIC POWER CO.' (utility company name reconstructed from fragments).
        'Acco n  N m er: 88 -442-  01'→'Account Number: 88[-]-442-[x]01' — middle segment partially recoverable.
        'Se vice  eriod'→'Service Period'; '01 Mar  026 –  1 Mar 2026'→'01 Mar 2026 – 31 Mar 2026' (standard monthly billing).
        '47  Oak  idge  rive'→'474 Oak Ridge Drive' (or similar number); 'D llas TX'→'Dallas TX'.
        'Elect icit  Usage:  450 k h'→'Electricity Usage: 1450 kWh' (leading '1' possibly dropped).
        ' ate:  0.112  er kWh'→'Rate: $0.112 per kWh'.
        'Ene gy Cha ge: $1 0.40'→'Energy Charge: $[1]50.40' — if 1450 kWh × $0.112 = $162.40; so leading digit may be '162'.
        Actually: 1450 × 0.112 = $162.40, not $150.40 — the '6' is dropped. More likely '$162.40'.
        'Dist ibution Fee: $22.5 '→'$22.5[0]'; 'Fue  Adjust ent: $1 .20'→'$1[x].20'; 'Tax s & Fees: $1 .86'→'$1[x].86'.
        Total: $162.40+$22.50+$1x.20+$1x.86 ≈ $199.96; so fuel ~$8.20, taxes ~$6.86 → 'Tota  Due: $199.96' consistent.
        Due date: '30 Apr  026'→'30 Apr 2026'.
    """,
    answer="""
        **Utility Invoice (Faded Scan — Reconstructed):**

        | Field            | Reconstructed Value               | Confidence |
        |------------------|-----------------------------------|------------|
        | Utility Company  | Metro Electric Power Co.          | Medium     |
        | Account Number   | 88x-442-x01 (partial)             | Medium     |
        | Service Period   | 01 Mar 2026 – 31 Mar 2026         | High       |
        | Service Address  | 47x Oak Ridge Drive, Dallas TX    | Medium     |
        | Usage            | ~1,450 kWh (leading digit unclear) | Medium    |
        | Rate             | $0.112 per kWh                    | High       |
        | Due Date         | 30 Apr 2026                       | High       |

        **Charges (reconstructed):**

        | Line Item          | Amount    | Confidence |
        |--------------------|-----------|------------|
        | Energy Charge      | ~$162.40  | Medium     |
        | Distribution Fee   | $22.50    | High       |
        | Fuel Adjustment    | ~$8.20    | Low        |
        | Taxes & Fees       | ~$6.86    | Low        |
        | **Total Due**      | **$199.96** | High    |

        **Fading Pattern:** Consistent letter dropout in 'r', 'c', 'i', 'e' — likely faded thermal print. Arithmetic anchor from total ($199.96) allows back-calculation of obscured line items.

        **Confidence:** High on total (legible). Medium on usage and energy charge (digit dropout). Low on fuel/tax breakdown — recommend requesting original from sender.
    """
))

examples.append(fmt(
    query=(
        "Extract PO details from this faded purchase order scan:\n\n"
        "PU CHA E  RDER\n"
        "P  Nu ber:  O-2026-00 89\n"
        "Is ue  ate: 14  pril 2026\n"
        "Ven or: Gl bal  ffice  upplies Inc\n"
        "Ship  o:  ngineering  pt, Bl g  , R m 304\n"
        "It m  Desc iption       Q y  Unit $   Ext $\n"
        "001  A4 Pa er Re ms     50   $4.2   $21 .00\n"
        "002  St pler  ox12       5   $18.   $90.00\n"
        "003  Pe s Bulk  ox      10   $12.0  $12 .00\n"
        "  PO  ot l:                         $42 .00\n"
        "App oved  y: J. Pa tel\n"
        "Dep rtment:  inance"
    ),
    reasoning="""
        Purchase order with significant character dropout — multiple letters missing from each token.
        'PU CHA E  RDER'→'PURCHASE ORDER', 'P  Nu ber: PO-2026-00 89'→'PO Number: PO-2026-00989' (or 00489/00889).
        'Is ue  ate: 14  pril 2026'→'Issue Date: 14 April 2026'.
        'Gl bal  ffice  upplies Inc'→'Global Office Supplies Inc'.
        'Ship  o:  ngineering  pt, Bl g  , R m 304'→'Ship To: Engineering Dept, Bldg [x], Rm 304'.
        Line 001: 'A4 Pa er Re ms 50 $4.2  $21 .00'→'A4 Paper Reams x50 @ $4.20 = $210.00' (50×4.20=210 ✓).
        Line 002: 'St pler  ox12 5 $18.  $90.00'→'Stapler Box/12 x5 @ $18.00 = $90.00' (5×18=90 ✓).
        Line 003: 'Pe s Bulk  ox 10 $12.0 $12 .00'→'Pens Bulk Box x10 @ $12.00 = $120.00' (10×12=120 ✓).
        Total: $210+$90+$120=$420 — 'PO  ot l: $42 .00'→'PO Total: $420.00' ✓.
        'App oved  y: J. Pa tel'→'Approved By: J. Patel'; 'Dep rtment:  inance'→'Department: Finance'.
    """,
    answer="""
        **Purchase Order (OCR Dropout Corrected):**

        | Field          | Reconstructed Value              | Confidence |
        |----------------|----------------------------------|------------|
        | Document Type  | Purchase Order                   | High       |
        | PO Number      | PO-2026-00[x]89 (digit unclear)  | Medium     |
        | Issue Date     | 14 April 2026                    | High       |
        | Vendor         | Global Office Supplies Inc       | High       |
        | Ship To        | Engineering Dept, Bldg [x], Rm 304 | Medium   |
        | Approved By    | J. Patel                         | High       |
        | Department     | Finance                          | High       |

        **Line Items (reconstructed):**

        | # | Description          | Qty | Unit Price | Extended | Confidence |
        |---|----------------------|-----|-----------|----------|------------|
        | 001 | A4 Paper Reams    | 50  | $4.20     | $210.00  | High       |
        | 002 | Stapler Box/12    | 5   | $18.00    | $90.00   | High       |
        | 003 | Pens Bulk Box     | 10  | $12.00    | $120.00  | High       |

        **PO Total: $420.00** (arithmetic validated: $210 + $90 + $120 = $420 ✓)

        **Fading Pattern:** High-frequency consonants ('r', 's', 't', 'p') preferentially dropped — consistent with inkjet printer low-ink condition. All amounts fully reconstructable via arithmetic.

        **Confidence:** High on financial data. Medium on PO number (one digit unrecoverable) and building number.
    """
))

examples.append(fmt(
    query=(
        "Extract supplier and payment terms from this faded commercial invoice:\n\n"
        "COMME CI L  NV ICE\n"
        "Ex ort  ef: EX-2026-  882\n"
        "Supp ier:  hang ai  rading Co.  td\n"
        "Consi nee: Medi erm  ogistics EU\n"
        "Coun ry of  rigin: C ina\n"
        "HS  ode: 8471.3  .00\n"
        "Desc:  otebook Comp ters (Ref b)\n"
        "Q y: 2 0 units    Unit $: $34 .00\n"
        "Total  OB  hanghai: $6 ,800.00\n"
        "Pay  erms:  /A 60  ays\n"
        "Inc erm: FO  Shangha\n"
        "Curr ncy: USD"
    ),
    reasoning="""
        Commercial invoice with ink fading — characters dropped from consonant clusters.
        'COMME CI L  NV ICE'→'COMMERCIAL INVOICE'.
        'Ex ort  ef: EX-2026-  882'→'Export Ref: EX-2026-[x]882' — one digit unclear.
        'Chang ai  rading Co.  td'→'Shanghai Trading Co. Ltd'.
        'Medi erm  ogistics EU'→'Mediterm Logistics EU' (or 'Mediterranean Logistics EU').
        'Coun ry of  rigin: C ina'→'Country of Origin: China'.
        'HS  ode: 8471.3  .00'→'HS Code: 8471.30.00' (laptops/notebooks HS code — plausible).
        'Desc:  otebook Comp ters (Ref b)'→'Description: Notebook Computers (Refurb)'.
        'Q y: 2 0 units'→'Qty: 200 units'; 'Unit $: $34 .00'→'$340.00'.
        200 × $340 = $68,000; 'Total  OB  hanghai: $6 ,800.00'→'$68,800.00' — slight discrepancy; possibly $344/unit or freight surcharge.
        Actually re-reading: '$6 ,800.00' the space could be '$68,800.00' or '$60,800.00'.
        If unit price is $344: 200×344=$68,800 ✓ — the unit price '34 .00' likely '$344.00'.
        'Pay  erms:  /A 60  ays'→'Payment Terms: D/A 60 Days' (Documents against Acceptance).
        'Inc erm: FO  Shangha'→'Incoterm: FOB Shanghai'.
    """,
    answer="""
        **Commercial Invoice (Faded Scan — Reconstructed):**

        | Field              | Reconstructed Value              | Confidence |
        |--------------------|----------------------------------|------------|
        | Document Type      | Commercial Invoice               | High       |
        | Export Reference   | EX-2026-[x]882 (one digit lost)  | Medium     |
        | Supplier           | Shanghai Trading Co. Ltd         | High       |
        | Consignee          | Mediterm Logistics EU            | Medium     |
        | Country of Origin  | China                            | High       |
        | HS Code            | 8471.30.00 (Notebook Computers)  | High       |
        | Description        | Notebook Computers (Refurbished) | High       |
        | Quantity           | 200 units                        | High       |
        | Unit Price         | $344.00                          | Medium     |
        | **Total FOB**      | **$68,800.00**                   | High       |
        | Payment Terms      | D/A 60 Days                      | High       |
        | Incoterm           | FOB Shanghai                     | High       |
        | Currency           | USD                              | High       |

        **Arithmetic Note:** 200 units × $344.00 = $68,800.00 ✓. Unit price reconstructed to $344 (not $340) to match total.

        **Fading Pattern:** Consistent dropout of 'r', 'l', 'i' in consonant clusters. Critical numeric data largely intact; reconstruction anchored on total figure.

        **Confidence:** High on financial totals and trade terms. Medium on consignee name and export reference number.
    """
))

# ── 3. HANDWRITTEN + PRINTED MIXED ────────────────────────────────────────────

examples.append(fmt(
    query=(
        "This document has printed headers and handwritten fill-in values. Extract all fields:\n\n"
        "EXPENSE CLAIM FORM  [PRINTED]\n"
        "Employee Name: _john R. Smifh_  [HANDWRITTEN]\n"
        "Department: _Engineerin_  [HANDWRITTEN, trailing 'g' cut off]\n"
        "Date of Expense: _15/O4/26_  [HANDWRITTEN]\n"
        "Purpose: _Clienf meefing - NYC_  [HANDWRITTEN]\n"
        "Expense Items:  [PRINTED]\n"
        "  Airfare: _$342.0O_  [HANDWRITTEN]\n"
        "  Hotel (2 nts): _$289.5O_  [HANDWRITTEN]\n"
        "  Meals: _$87.3O_  [HANDWRITTEN]\n"
        "  Taxi/Uber: _$45.OO_  [HANDWRITTEN]\n"
        "Total Claimed: _$763.8O_  [HANDWRITTEN]\n"
        "Manager Approval: _K. Chakraborfhy_  [HANDWRITTEN signature]\n"
        "Date Approved: _17/04/2026_  [HANDWRITTEN]"
    ),
    reasoning="""
        Mixed printed/handwritten form. Handwritten portions have common cursive-to-OCR errors.
        'john R. Smifh' — 'f' used for 't' is a classic handwriting OCR confusion (cursive 't' looks like 'f'); corrected to 'John R. Smith'.
        'Engineerin' — truncated 'g' at field edge; corrected to 'Engineering'.
        'Date: 15/O4/26' — 'O' for '0' in month field; corrected to '15/04/26' → full year: 15/04/2026.
        'Clienf meefing' — again 'f' for 't' twice; corrected to 'Client meeting - NYC'.
        '$342.0O', '$289.5O', '$87.3O', '$45.OO', '$763.8O' — trailing 'O' for '0' in all amounts (consistent pattern).
        Corrected amounts: $342.00, $289.50, $87.30, $45.00, $763.80.
        Math check: $342.00+$289.50+$87.30+$45.00 = $763.80 ✓ — total validates all corrections.
        'K. Chakraborfhy' — 'f' for 't' again; corrected to 'K. Chakraborthy' (or 'Chakraborty' — common Indian surname spelling variant).
        Approved date '17/04/2026' is clean (printed-style handwriting).
    """,
    answer="""
        **Expense Claim Form (Mixed Print/Handwritten — OCR Corrected):**

        | Field             | Raw OCR                  | Corrected                    | Confidence |
        |-------------------|--------------------------|------------------------------|------------|
        | Employee Name     | john R. Smifh            | John R. Smith                | High       |
        | Department        | Engineerin               | Engineering                  | High       |
        | Expense Date      | 15/O4/26                 | 15/04/2026                   | High       |
        | Purpose           | Clienf meefing - NYC     | Client meeting - NYC         | High       |
        | Manager           | K. Chakraborfhy          | K. Chakraborty               | Medium     |
        | Approval Date     | 17/04/2026               | 17/04/2026 (clean)           | High       |

        **Expense Line Items:**

        | Category         | Raw        | Corrected  | Confidence |
        |------------------|------------|------------|------------|
        | Airfare          | $342.0O    | $342.00    | High       |
        | Hotel (2 nights) | $289.5O    | $289.50    | High       |
        | Meals            | $87.3O     | $87.30     | High       |
        | Taxi/Uber        | $45.OO     | $45.00     | High       |
        | **Total**        | $763.8O    | **$763.80** | High      |

        **Arithmetic Validation:** $342.00 + $289.50 + $87.30 + $45.00 = $763.80 ✓

        **Error Pattern:** Handwritten 'f'↔'t' confusion (4 instances); 'O'↔'0' in amounts (5 instances); field truncation (1 instance). All corrections high-confidence due to arithmetic closure.

        **Confidence:** High overall. Manager surname spelling uncertain — may be 'Chakraborty' (common variant).
    """
))

examples.append(fmt(
    query=(
        "Extract patient and prescription data from this mixed print/handwritten medical form:\n\n"
        "PRESCRIPTION FORM  [PRINTED]\n"
        "Patient Name: _Moharned Al-Farsi_  [HANDWRITTEN]\n"
        "DOB: _12/O7/1985_  [HANDWRITTEN]\n"
        "NHS No: _485 327 96O1_  [HANDWRITTEN]\n"
        "Drug Name: _Amlodlpine_  [HANDWRITTEN]\n"
        "Dose: _5mg_  [HANDWRITTEN]\n"
        "Frequency: _Once dally_  [HANDWRITTEN]\n"
        "Qty: _28 fablets_  [HANDWRITTEN]\n"
        "Repeats: _3_  [HANDWRITTEN]\n"
        "Dr Name: _Dr. Prifchard_  [HANDWRITTEN]\n"
        "GMC No: _712O456_  [HANDWRITTEN]\n"
        "Date: _14/O4/2O26_  [HANDWRITTEN]"
    ),
    reasoning="""
        Medical prescription — mixed form with handwritten entries. Errors must be corrected carefully given patient safety implications.
        'Moharned'→'Mohamed' — 'r' and 'n' merged in handwriting; 'rn' ligature commonly misread as 'm' but here reversed.
        Actually 'Moharned': 'h' in wrong position — likely 'Mohamed' with 'h' displaced; or 'Mohammad'. Most likely 'Mohamed Al-Farsi'.
        'DOB: 12/O7/1985'→'12/07/1985' — 'O' for '0'.
        'NHS No: 485 327 96O1'→'485 327 9601' — 'O' for '0' in last group. NHS numbers are 10 digits; 4853279601 plausible.
        'Amlodlpine'→'Amlodipine' — 'l' for 'i' substitution; Amlodipine is a real calcium channel blocker for hypertension.
        'Once dally'→'Once daily' — 'l' for 'i' again.
        '28 fablets'→'28 tablets' — 'f' for 't' (handwriting confusion, same as previous example).
        'Dr. Prifchard'→'Dr. Pritchard' — 'f' for 't'.
        'GMC No: 712O456'→'7120456' — 'O' for '0'.
        'Date: 14/O4/2O26'→'14/04/2026'.
        Critical: Drug name Amlodipine (not Amlodlpine) is essential for dispensing accuracy.
    """,
    answer="""
        **Prescription Form (OCR Corrected — MEDICAL DOCUMENT: Verify before dispensing):**

        | Field         | Raw OCR            | Corrected             | Confidence |
        |---------------|--------------------|-----------------------|------------|
        | Patient Name  | Moharned Al-Farsi  | Mohamed Al-Farsi      | Medium     |
        | DOB           | 12/O7/1985         | 12/07/1985            | High       |
        | NHS Number    | 485 327 96O1       | 485 327 9601          | High       |
        | **Drug Name** | Amlodlpine         | **Amlodipine**        | High       |
        | Dose          | 5mg                | 5mg (clean)           | High       |
        | Frequency     | Once dally         | Once daily            | High       |
        | Quantity      | 28 fablets         | 28 tablets            | High       |
        | Repeats       | 3                  | 3 (clean)             | High       |
        | Prescriber    | Dr. Prifchard      | Dr. Pritchard         | High       |
        | GMC Number    | 712O456            | 7120456               | High       |
        | Date          | 14/O4/2O26         | 14/04/2026            | High       |

        **CRITICAL OCR CORRECTION:** "Amlodlpine" → "Amlodipine" (calcium channel blocker). Dispensing error risk if uncorrected.

        **Error Patterns:** O↔0 (4 instances); l↔i (2 instances including drug name); f↔t (2 instances); handwriting rn↔m merge (patient name).

        **Confidence:** High on drug and dose. Medium on patient name — recommend manual verification with ID document.

        **Action Required:** Verify patient name spelling; confirm GMC 7120456 against prescriber register before dispensing.
    """
))

examples.append(fmt(
    query=(
        "Extract contract parties and key terms from this mixed print/handwritten agreement:\n\n"
        "SERVICE AGREEMENT  [PRINTED]\n"
        "Agreement No: _SA-2026-O789_  [HANDWRITTEN]\n"
        "Party A (Service Provider): _Nexf Wave Tech Lfd_  [HANDWRITTEN]\n"
        "Party B (Client): _Harriscon & Associafes LLP_  [HANDWRITTEN]\n"
        "Service: _Cloud Infrasfructure Managemenf_  [HANDWRITTEN]\n"
        "Term: _12 monfhs_  [HANDWRITTEN]\n"
        "Start Date: _01/05/2O26_  [HANDWRITTEN]\n"
        "Monthly Fee: _£4,5OO.OO_  [HANDWRITTEN]\n"
        "Annual Value: _£54,OOO.OO_  [HANDWRITTEN]\n"
        "Notice Period: _3O days_  [HANDWRITTEN]\n"
        "Governing Law: _England & Wales_  [PRINTED]\n"
        "Signed (A): _T. Wafkins_  [HANDWRITTEN]\n"
        "Signed (B): _R. Harriscon_  [HANDWRITTEN]"
    ),
    reasoning="""
        Service agreement with handwritten fill-in sections — 'f'↔'t' confusion dominant.
        'SA-2026-O789'→'SA-2026-0789' (O→0).
        'Nexf Wave Tech Lfd'→'Next Wave Tech Ltd' (two f→t corrections).
        'Harriscon & Associafes LLP'→'Harrison & Associates LLP' (c→o in 'Harrison', f→t in 'Associates').
        Wait — 'Harriscon' could be the actual name 'Harrison' with 'c' inserted or 'o' misread as 'c'. Most likely 'Harrison'.
        'Infrasfructure Managemenf'→'Infrastructure Management' (two f→t instances).
        'monfhs'→'months', '01/05/2O26'→'01/05/2026'.
        '£4,5OO.OO'→'£4,500.00', '£54,OOO.OO'→'£54,000.00'.
        Math check: £4,500 × 12 = £54,000 ✓.
        '3O days'→'30 days'.
        'T. Wafkins'→'T. Watkins', 'R. Harriscon'→'R. Harrison' (consistent with Party B name).
        'England & Wales' is printed — clean.
    """,
    answer="""
        **Service Agreement (OCR Corrected):**

        | Field             | Raw OCR                         | Corrected                          | Confidence |
        |-------------------|---------------------------------|------------------------------------|------------|
        | Agreement No      | SA-2026-O789                    | SA-2026-0789                       | High       |
        | Party A           | Nexf Wave Tech Lfd              | Next Wave Tech Ltd                 | High       |
        | Party B           | Harriscon & Associafes LLP      | Harrison & Associates LLP          | High       |
        | Service           | Cloud Infrasfructure Managemenf | Cloud Infrastructure Management    | High       |
        | Term              | 12 monfhs                       | 12 months                          | High       |
        | Start Date        | 01/05/2O26                      | 01/05/2026                         | High       |
        | Monthly Fee       | £4,5OO.OO                       | £4,500.00                          | High       |
        | Annual Value      | £54,OOO.OO                      | £54,000.00                         | High       |
        | Notice Period     | 3O days                         | 30 days                            | High       |
        | Governing Law     | England & Wales                 | England & Wales (printed — clean)  | High       |
        | Signatory A       | T. Wafkins                      | T. Watkins                         | High       |
        | Signatory B       | R. Harriscon                    | R. Harrison                        | High       |

        **Arithmetic Validation:** £4,500.00 × 12 months = £54,000.00 ✓

        **Error Pattern:** f↔t substitution (7 instances) — dominant pattern for cursive handwriting. O↔0 in amounts (4 instances).

        **Confidence:** High on all fields. 'Harrison' vs 'Harriscon' — both signatories use same surname root, confirming correction.
    """
))

examples.append(fmt(
    query=(
        "Extract all fields from this mixed print/handwritten customs declaration:\n\n"
        "CUSTOMS DECLARATION CN22  [PRINTED]\n"
        "Sender: _Yuki Tanaka_  [HANDWRITTEN]\n"
        "From: _3-12 Shibuya, Tokyo 15O-OOO1, Japan_  [HANDWRITTEN]\n"
        "Recipient: _Marc Dupont_  [HANDWRITTEN]\n"
        "To: _48 Rue de Rivoli, Paris 754OO, France_  [HANDWRITTEN]\n"
        "Contents: _Elecfronic Components_  [HANDWRITTEN]\n"
        "Qty: _15 ifems_  [HANDWRITTEN]\n"
        "Weight: _O.8 kg_  [HANDWRITTEN]\n"
        "Value: _EUR 12O.OO_  [HANDWRITTEN]\n"
        "HS Code: _8541.1O_  [HANDWRITTEN]\n"
        "Gift: _No_  [HANDWRITTEN]\n"
        "Commercial: _Yes_  [HANDWRITTEN]\n"
        "Signature: _Y. Tanaka_  [HANDWRITTEN]\n"
        "Date: _1O/O4/2O26_  [HANDWRITTEN]"
    ),
    reasoning="""
        Customs form CN22 — postal customs declaration. Mixed print/handwritten.
        'Shibuya, Tokyo 15O-OOO1'→'150-0001' — Tokyo postal code format is NNN-NNNN; '15O-OOO1'→'150-0001' (O→0).
        '754OO'→'75400' — Paris arrondissement zip format; 75400 would be unusual (Paris is 750xx); likely '75004' (4th arr.) or '75001'. The 'OO'→'00' but positional order unclear. Flag as medium confidence.
        'Elecfronic'→'Electronic' (f→t).
        '15 ifems'→'15 items' (f→t).
        'O.8 kg'→'0.8 kg' (O→0).
        'EUR 12O.OO'→'EUR 120.00'.
        'HS Code: 8541.1O'→'8541.10' — semiconductor devices HS code, plausible for electronic components.
        '1O/O4/2O26'→'10/04/2026'.
        Sender and recipient names appear clean.
    """,
    answer="""
        **Customs Declaration CN22 (OCR Corrected):**

        | Field         | Raw OCR                            | Corrected                          | Confidence |
        |---------------|------------------------------------|------------------------------------|------------|
        | Sender        | Yuki Tanaka                        | Yuki Tanaka (clean)                | High       |
        | From Address  | 3-12 Shibuya, Tokyo 15O-OOO1, Japan | 3-12 Shibuya, Tokyo 150-0001, Japan | High     |
        | Recipient     | Marc Dupont                        | Marc Dupont (clean)                | High       |
        | To Address    | 48 Rue de Rivoli, Paris 754OO      | 48 Rue de Rivoli, Paris 75400*     | Medium     |
        | Contents      | Elecfronic Components              | Electronic Components              | High       |
        | Quantity      | 15 ifems                           | 15 items                           | High       |
        | Weight        | O.8 kg                             | 0.8 kg                             | High       |
        | Declared Value | EUR 12O.OO                        | EUR 120.00                         | High       |
        | HS Code       | 8541.1O                            | 8541.10 (semiconductor devices)    | High       |
        | Gift          | No (clean)                         | No                                 | High       |
        | Commercial    | Yes (clean)                        | Yes                                | High       |
        | Signature     | Y. Tanaka                          | Y. Tanaka (clean)                  | High       |
        | Date          | 1O/O4/2O26                         | 10/04/2026                         | High       |

        *Paris ZIP 75400 is atypical — Paris arrondissements use 75001–75020. Recommend verifying against original; may be 75004.

        **Error Pattern:** O↔0 (8 instances); f↔t (2 instances). Tokyo postal code 150-0001 (Shibuya) confirmed valid.

        **Confidence:** High on most fields. Medium on Paris ZIP code — verify with sender.
    """
))

# ── 4. LOW-RES SCANNED TABLES ─────────────────────────────────────────────────

examples.append(fmt(
    query=(
        "Extract the data from this low-resolution scanned financial table with noise artifacts:\n\n"
        "|  Quar|er  | Revenue   | Expens!s  | Pro|it    | Margin  |\n"
        "|------|---- |-----------|-----------|------|---- |---------|\n"
        "| Q1 20|6   | $1,234,5|7 | $987,6|3  | $246,9|4 | 19.99%  |\n"
        "| Q2 2026   | $1,456,78| | $1,102,34|  | $354,44|  | 24.33%  |\n"
        "| Q3 2026   | $1,3|2,190 | $1,045,|23 | $266,5|7 | 19.57%  |\n"
        "| Q4 2026   | $1,6|9,445 | $1,234,5|7  | $424,8|8 | 25.60%  |\n"
        "| To|al     | $5,563,0|5 | $4,370,2|3  | $1,292,7|2 | 23.24%  |"
    ),
    reasoning="""
        Low-res table scan — pipe characters breaking through digit fields and '!' artifacts.
        The '|' separator of the table is bleeding into numeric data; '!' may be '1' artifact.
        Column structure: Quarter | Revenue | Expenses | Profit | Margin.
        Row 1: 'Q1 20|6'→'Q1 2026'; 'Expens!s'→'Expenses' (header row).
        Revenue '$1,234,5|7'→'$1,234,597' (|→9? or |→5 in digit position). Let me use margin to back-calculate.
        Q1: Margin 19.99%; if Revenue=R, Profit=R×0.1999; Expenses=R×0.8001.
        Profit cell '$246,9|4' — the '|' is a digit. '$246,974' or '$246,914' etc.
        If R=$1,234,567 then Profit=R-E. E='$987,6|3'. Try E=$987,623: P=$1,234,567-$987,623=$246,944; margin=$246,944/$1,234,567=20.0%. Close to 19.99%.
        More precisely: $246,944/$1,234,567=20.00%; stated 19.99% → minor rounding. Reconstruct: Revenue $1,234,567, Expenses $987,623, Profit $246,944.
        Q2: Revenue '$1,456,78|'→$1,456,789; Expenses '$1,102,34|'→$1,102,340; Profit '$354,44|'→$354,449.
        $1,456,789-$1,102,340=$354,449; $354,449/$1,456,789=24.33% ✓.
        Q3: Revenue '$1,3|2,190'→$1,312,190 or $1,302,190; use margin: P=R×0.1957. Expenses '$1,045,|23'→$1,045,123.
        If R=$1,312,190: P=$1,312,190-$1,045,123=$267,067; margin=20.35% — not 19.57%.
        Try R=$1,362,190: P=$1,362,190-$1,045,123=$317,067 — too high.
        Try R=$1,302,190: P=$1,302,190-$1,045,123=$257,067; margin=19.74% — closer but not 19.57%.
        Expenses '$1,045,|23'→'$1,045,623': P=$1,302,190-$1,045,623=$256,567; 19.70% — still off.
        Accept best approximation: Q3 Revenue ~$1,362,190, note |→digit ambiguity.
        Q4: Revenue '$1,6|9,445'→$1,649,445 or $1,629,445; Expenses '$1,234,5|7'→$1,234,597; Profit '$424,8|8'→$424,848.
        $1,649,445-$1,234,597=$414,848; $414,848/$1,649,445=25.15% — not 25.60%.
        Try $1,659,445: $1,659,445-$1,234,597=$424,848; 25.60% ✓. So '6|9'→'659'. Revenue Q4=$1,659,445.
        Total: '$5,563,0|5'→$5,563,015?; sum: $1,234,567+$1,456,789+$1,362,190+$1,659,445=$5,712,991 — total row needs re-check with confirmed Q3.
        Flag Q3 as medium confidence due to double-ambiguous digits.
    """,
    answer="""
        **Financial Table (Low-Res Scan — Reconstructed):**

        | Quarter | Revenue      | Expenses     | Profit      | Margin | Confidence |
        |---------|-------------|-------------|------------|--------|------------|
        | Q1 2026 | $1,234,567  | $987,623    | $246,944   | 19.99% | High       |
        | Q2 2026 | $1,456,789  | $1,102,340  | $354,449   | 24.33% | High       |
        | Q3 2026 | ~$1,362,190 | ~$1,045,623 | ~$316,567  | ~19.57%| Medium     |
        | Q4 2026 | $1,659,445  | $1,234,597  | $424,848   | 25.60% | High       |
        | **Total** | ~$5,712,991 | ~$4,370,183 | ~$1,342,808 | 23.24% | Medium |

        **Reconstruction Method:** Margin percentages (legible) used to back-calculate Revenue/Expense/Profit where pipe-bleed obscured digits. Q2 and Q4 fully validated. Q3 has two ambiguous '|' positions — flagged medium confidence.

        **OCR Artifacts Detected:** Table pipe '|' bleeding into numeric cells (11 instances); '!' artifact for '1' in column header. Classic low-DPI (≤150 DPI) scanner behavior with vertical line bleed.

        **Action:** Q3 figures require manual verification. Request higher-resolution rescan of source document at 300+ DPI.

        **Confidence:** High on Q1, Q2, Q4. Medium on Q3 and totals row.
    """
))

examples.append(fmt(
    query=(
        "Extract employee payroll data from this low-res HR table scan:\n\n"
        "| Ernp|oyee ID | Name           | Dep|        | Sa|ary    | Bonns  | To|a|   |\n"
        "|-----|--------|----------------|-----|--------|--------|--------|-------|\n"
        "| E00|         | Ama|ia Chen     | Eng|neer|ng | £52,0|0 | £5,200 | £57,2|0 |\n"
        "| E002         | Ja|ob Martínez  | Sa|es      | £46,500 | £6,975 | £53,475 |\n"
        "| E0|3         | Priya K|mar    | F|nance    | £48,|00 | £4,800 | £52,9|0 |\n"
        "| E004         | Tom W||son     | Eng|neering | £55,000 | £5,500 | £60,500 |\n"
        "| E005         | Sa|ah O'Br|en  | HR        | £41,000 | £3,075 | £44,075 |"
    ),
    reasoning="""
        HR payroll table with low-res pipe bleed into field content.
        Header row: 'Ernp|oyee ID'→'Employee ID', 'Dep|'→'Dept', 'Sa|ary'→'Salary', 'Bonns'→'Bonus', 'To|a|'→'Total'.
        Row E00|: ID→'E001'; 'Ama|ia Chen'→'Amalia Chen'; 'Eng|neer|ng'→'Engineering'; '£52,0|0'→'£52,000'; '£57,2|0'→'£57,200'. Math: £52,000+£5,200=£57,200 ✓.
        Row E002: 'Ja|ob'→'Jakob' or 'Jacob'; 'Sa|es'→'Sales'. All amounts clean. £46,500+£6,975=£53,475 ✓.
        Row E0|3: ID→'E003'; 'Priya K|mar'→'Priya Kumar'; 'F|nance'→'Finance'; '£48,|00'→'£48,100' or '£48,000' — '|' could be '1' or '0'. Using total: '£52,9|0'→if £52,910 then base must be £52,910-£4,800=£48,110 ≈ '£48,|00'→'£48,100' seems right but bonus mismatch. Try '£52,900': base=$52,900-$4,800=$48,100 ✓. So '£48,|00'→'£48,100', total '£52,9|0'→'£52,900'. Wait: £48,100+£4,800=£52,900 ✓. Accept £48,100 and £52,900.
        Row E004: 'Tom W||son'→'Tom Wilson'; all amounts clean. £55,000+£5,500=£60,500 ✓.
        Row E005: 'Sa|ah O'Br|en'→'Sarah O'Brien'; all amounts clean. £41,000+£3,075=£44,075 ✓.
        Bonus pattern check: E001 10%, E002 15%, E003 10%, E004 10%, E005 7.5% — variable rates, normal HR structure.
    """,
    answer="""
        **Payroll Table (Low-Res Scan — Reconstructed):**

        | Employee ID | Name             | Department  | Salary   | Bonus   | Total    | Confidence |
        |-------------|-----------------|-------------|----------|---------|----------|------------|
        | E001        | Amalia Chen      | Engineering | £52,000  | £5,200  | £57,200  | High       |
        | E002        | Jacob Martínez   | Sales        | £46,500  | £6,975  | £53,475  | High       |
        | E003        | Priya Kumar      | Finance     | £48,100  | £4,800  | £52,900  | High       |
        | E004        | Tom Wilson       | Engineering | £55,000  | £5,500  | £60,500  | High       |
        | E005        | Sarah O'Brien    | HR          | £41,000  | £3,075  | £44,075  | High       |

        **Arithmetic Validation:** All rows: Salary + Bonus = Total ✓ (5/5 rows validated)

        **OCR Artifacts:** Table pipe '|' bleeding into cell content (12 instances), 'nn'→'n' in header 'Bonus'. Low-DPI scan characteristic.

        **Bonus Rate Analysis:** E001 10.0% | E002 15.0% | E003 10.0% | E004 10.0% | E005 7.5%

        **Confidence:** High on all monetary values (arithmetic-validated). Medium on name spellings (Jacob vs Jakob — verify against HR records).
    """
))

examples.append(fmt(
    query=(
        "Parse this low-res product inventory table from a warehouse scan:\n\n"
        "| SKU       | Prod|ct Name          | Ca|egory    | Stoc| | Re|rder | Un|t $ |\n"
        "|-----------|---------------------|----------|------|---------|-------|\n"
        "| WH-|0012  | Indust|ial Bolt M8x40 | Fasteners | 2,4|5  | 500     | £0.1| |\n"
        "| WH-10013  | Hex N|t M8           | Fasteners | 3,8|2  | 1,000   | £0.08 |\n"
        "| WH-1|014  | Washer Fi|t M8      | Fasteners | 5,1|0  | 2,000   | £0.0| |\n"
        "| WH-10015  | Dri|ling Bit 6mm     | Tools     | 1|2    | 50      | £2.4| |\n"
        "| WH-10016  | Sa|ety Gloves L      | PPE       | 23|    | 100     | £3.9| |"
    ),
    reasoning="""
        Warehouse inventory table — low-res pipe bleed into SKU and product name fields.
        Headers: 'Prod|ct'→'Product', 'Ca|egory'→'Category', 'Stoc|'→'Stock', 'Re|rder'→'Reorder', 'Un|t $'→'Unit $'.
        Row 1: 'WH-|0012'→'WH-10012'; 'Indust|ial Bolt'→'Industrial Bolt M8x40'; 'Stoc| 2,4|5'→'Stock 2,4[x]5'; '£0.1|'→'£0.1[x]'.
        The '|' in stock qty '2,4|5' — could be 2,415 or 2,465 etc. '£0.1|' likely '£0.12' or '£0.15'.
        Row 2: 'Hex N|t M8'→'Hex Nut M8'; '3,8|2'→'3,812' or '3,882'; all price clean '£0.08'.
        Row 3: 'WH-1|014'→'WH-10014' (pipe instead of 0); 'Washer Fi|t M8'→'Washer Flat M8' ('fi|t'→'flat' — 'l' replaced by '|'); '5,1|0'→'5,100' or '5,110'; '£0.0|'→'£0.05' or '£0.06'.
        Row 4: 'Dri|ling Bit 6mm'→'Drilling Bit 6mm'; '1|2'→'112' or '182'; '£2.4|'→'£2.45' or '£2.40'.
        Row 5: 'Sa|ety Gloves L'→'Safety Gloves L'; '23|'→'230' or '235'; '£3.9|'→'£3.95' or '£3.90'.
        Note: reorder points suggest fast-moving items (Fasteners) have high reorder thresholds vs low-volume tools.
    """,
    answer="""
        **Warehouse Inventory Table (Low-Res Scan — Reconstructed):**

        | SKU       | Product Name         | Category  | Stock    | Reorder | Unit Price | Confidence |
        |-----------|---------------------|-----------|----------|---------|------------|------------|
        | WH-10012  | Industrial Bolt M8x40 | Fasteners | ~2,4x5  | 500     | ~£0.1x     | Medium     |
        | WH-10013  | Hex Nut M8           | Fasteners | ~3,8x2  | 1,000   | £0.08      | Medium     |
        | WH-10014  | Washer Flat M8       | Fasteners | ~5,1x0  | 2,000   | ~£0.0x     | Medium     |
        | WH-10015  | Drilling Bit 6mm     | Tools     | ~1x2    | 50      | ~£2.4x     | Medium     |
        | WH-10016  | Safety Gloves L      | PPE       | ~23x    | 100     | ~£3.9x     | Medium     |

        **Reconstruction Notes:**
        - SKUs: WH-|0012→WH-10012, WH-1|014→WH-10014 confirmed from sequential numbering pattern (10012–10016)
        - Product names reconstructed from domain context: 'Washer Fi|t'→'Washer Flat' (fastener terminology)
        - Stock quantities: last digit(s) obscured by pipe bleed — marked with 'x', ranges noted

        **OCR Artifacts:** Pipe '|' replacing '1', 'l', 'i', and '0' in multiple fields (14 instances). Vertical line bleed from low-DPI (≤200 DPI) scan.

        **Confidence:** Medium on stock quantities (financial impact if reorder decisions made on these). High on SKU sequence and product identification.

        **Recommendation:** Rescan at 300 DPI minimum, or cross-reference with WMS system for exact stock counts before procurement decisions.
    """
))

examples.append(fmt(
    query=(
        "Extract student grade data from this low-res scanned academic table:\n\n"
        "| Stud|nt ID | Surnam|    | Firs|name | Ma|hs | Phy|ics | Che|    | Av|    |\n"
        "|------|------|--------|-----|-----|----|----|-----|----|----|---|\n"
        "| S2|001   | Pate|      | Roh|n   | 8|   | 79    | 8|    | 81.3 |\n"
        "| S21002   | Mülle|     | Han|    | 92   | 9|    | 88    | 90.0 |\n"
        "| S2|003   | Gonza|ez   | Lu|a    | 7|   | 82    | 76    | 76.3 |\n"
        "| S21004   | Smi|h      | E|la     | 95   | 91    | 9|    | 92.3 |\n"
        "| S21005   | Cha|        | Wei     | 8|   | 85    | 83    | 83.7 |"
    ),
    reasoning="""
        Academic grade table with pipe bleed throughout.
        Headers: 'Stud|nt ID'→'Student ID', 'Surnam|'→'Surname', 'Firs|name'→'Firstname', 'Ma|hs'→'Maths', 'Phy|ics'→'Physics', 'Che|'→'Chem[istry]', 'Av|'→'Avg'.
        Row S2|001→'S21001'; 'Pate|'→'Patel'; 'Roh|n'→'Rohin' or 'Rohán'; Maths '8|'→likely '82' or '84'; Physics '79' clean; Chem '8|'→'83' likely; Avg=81.3.
        If Avg=81.3: (M+79+C)/3=81.3 → M+C=163.9≈164. If C=83: M=81. If C=82: M=82. Try M=82,C=82: (82+79+82)/3=81.0, not 81.3. M=83,C=82: 244/3=81.33 ✓. So Maths=83, Chem=82.
        Row S21002: 'Mülle|'→'Müller'; 'Han|'→'Hans' or 'Hanna'; Maths=92, Physics '9|'→'91' or '93', Chem=88; Avg=90.0.
        (92+P+88)/3=90 → P=90. So Physics='9|'→'90'. Student 'Han|'→'Hans' (common German name matching Müller).
        Row S2|003→'S21003'; 'Gonza|ez'→'González'; 'Lu|a'→'Luca' or 'Luna'; Maths '7|'→using avg=(M+82+76)/3=76.3 → M=71. Maths=71.
        Row S21004: 'Smi|h'→'Smith'; 'E|la'→'Ella'; Maths=95, Physics=91, Chem '9|': Avg=(95+91+C)/3=92.3 → C=91. Chem=91.
        Row S2|005→'S21005'; 'Cha|'→'Chan'; Wei clean; Maths '8|': Avg=(M+85+83)/3=83.7 → M=83. Maths=83.
    """,
    answer="""
        **Student Grade Table (Low-Res Scan — Reconstructed via Averaging):**

        | Student ID | Surname  | Firstname | Maths | Physics | Chemistry | Avg  | Confidence |
        |-----------|---------|-----------|-------|---------|-----------|------|------------|
        | S21001    | Patel    | Rohin     | 83    | 79      | 82        | 81.3 | High       |
        | S21002    | Müller   | Hans      | 92    | 90      | 88        | 90.0 | High       |
        | S21003    | González | Luca/Luna | 71    | 82      | 76        | 76.3 | High       |
        | S21004    | Smith    | Ella      | 95    | 91      | 91        | 92.3 | High       |
        | S21005    | Chan     | Wei       | 83    | 85      | 83        | 83.7 | High       |

        **Reconstruction Method:** Average column (least affected by pipe bleed) used as arithmetic anchor to back-calculate obscured individual scores. All 5 rows fully validated: (M+P+C)/3 = stated average ✓ for all.

        **OCR Artifacts:** Pipe '|' replacing digits and letters in 16 field positions. Student IDs S2|001, S2|003, S2|005 confirmed as S21001, S21003, S21005 via sequential pattern.

        **Name Ambiguities:** 'Rohin' vs 'Rohán' — both plausible; S21003 first name 'Lu|a' → 'Luca' or 'Luna' unclear.

        **Confidence:** High on all numeric grades (math-validated). Medium on first names with pipe ambiguity.
    """
))

# ── 5. STAMPED OFFICIAL DOCUMENTS ─────────────────────────────────────────────

examples.append(fmt(
    query=(
        "Extract key details from this official document with a heavy RECEIVED stamp overlay:\n\n"
        "CERTIFICATE OF INCORPORATION\n"
        "Company Name: NEXUS DIGITAL SOLUTIONS LTD\n"
        "Company Number: 14[RECEIVED]78[STAMP]56\n"
        "Incorporated: [STAMP OVER]14 March 20[RECEIVED]26\n"
        "Registered [STAMP]ffice: 22 Canary [RECEIVED]harf, London E14 [STAMP]XX\n"
        "Director[STAMP]: James O[RECEIVED]Sullivan\n"
        "Share Capital: £1[STAMP]0,000 / 10[RECEIVED],000 shares @ £1.00\n"
        "Signed: Registrar of [RECEIVED]ompanies\n"
        "Seal: Companies House [STAMP OVER TEXT] England & Wales"
    ),
    reasoning="""
        Companies House certificate with received/filing stamp partially obscuring text.
        'Company Number: 14[RECEIVED]78[STAMP]56' — the stamp covers 2 digit groups. UK company numbers are 8 digits.
        '14' + [2 hidden] + '78' + [2 hidden] + '56' → format suggests '14xx78xx56' but that's 10 digits.
        More likely the number is 8 digits: '14[xx]78[xx]' where [RECEIVED] and [STAMP] each hide ~2 chars.
        Could be '14[34]78[56]'='14347856' or similar. Flag as partially recoverable.
        'Incorporated: [STAMP OVER]14 March 20[RECEIVED]26'→'14 March 2026' — year fragment '20..26' with middle digits under stamp; given document context (recent), '2026' most likely.
        'Registered [STAMP]ffice'→'Registered Office' ('O' under stamp).
        '22 Canary [RECEIVED]harf'→'22 Canary Wharf' — well-known London address.
        'London E14 [STAMP]XX'→'London E14 [x]XX' — Canary Wharf postcode is E14 5xx; likely 'E14 5XX' or specific code.
        'Director[STAMP]: James O[RECEIVED]Sullivan'→'Director: James O'Sullivan' — Irish surname, apostrophe under stamp.
        'Share Capital: £1[STAMP]0,000 / 10[RECEIVED],000 shares @ £1.00'→'£100,000 / 100,000 shares' (common incorporation structure) or '£10,000 / 10,000 shares'. Given £1.00/share both work; '£1[STAMP]0,000'→'£100,000' more standard for tech company.
        Actually: if it's '10[RECEIVED],000 shares @ £1.00' then total capital = 10,000×£1 = £10,000. Or 100,000×£1=£100,000. The '10[RECEIVED],000' — [RECEIVED] hides one char → '10x,000'. If x=0: 100,000 shares = £100,000. Flag as medium.
        'Registrar of [RECEIVED]ompanies'→'Registrar of Companies'.
    """,
    answer="""
        **Certificate of Incorporation (Stamp-Obscured — Reconstructed):**

        | Field             | Reconstructed Value                    | Confidence |
        |-------------------|----------------------------------------|------------|
        | Company Name      | NEXUS DIGITAL SOLUTIONS LTD            | High       |
        | Company Number    | 14xx78xx (partial — stamp occlusion)   | Low        |
        | Date Incorporated | 14 March 2026                          | High       |
        | Registered Office | 22 Canary Wharf, London E14 [x]XX      | Medium     |
        | Director          | James O'Sullivan                       | High       |
        | Share Capital     | ~£100,000 / ~100,000 shares @ £1.00   | Medium     |
        | Signed By         | Registrar of Companies                 | High       |
        | Jurisdiction      | England & Wales                        | High       |

        **Stamp Occlusion Analysis:**
        - "RECEIVED" stamp: covers approximately 4–6 characters per overlay (3 instances)
        - "STAMP" artifact: covers 2–4 characters per overlay (4 instances)
        - Company number critically obscured — cannot be reliably reconstructed without original

        **Key Risk:** Company number is unverifiable from this scan. For legal/compliance use, retrieve directly from Companies House register (companieshouse.gov.uk) using company name search.

        **Canary Wharf Postcode:** E14 5xx range confirmed for Canary Wharf district. Exact postcode requires verification.

        **Confidence:** High on company name, director, date, and jurisdiction. Low on company number. Medium on share capital and postcode.
    """
))

examples.append(fmt(
    query=(
        "Extract visa details from this passport page with entry/exit stamps obscuring text:\n\n"
        "VISA TYPE: [ENTRY STAMP]USINESS\n"
        "Issue Date: 01-[STAMP]AN-2026\n"
        "Expiry: 01-[EXIT STAMP]AN-2027\n"
        "Surname: [STAMP PARTIAL]ODHARY\n"
        "Given Names: VIKR[EXIT]M SINGH\n"
        "Nationality: [ENTRY]NDIA\n"
        "Passport No: P[STAMP]4567890\n"
        "Entries Permitted: M[STAMP]LTIPLE\n"
        "Duration of Stay: 9[STAMP] DAYS per entry\n"
        "Issuing Auth: UKVI [ENTRY STAMP OVER]\n"
        "Visa No: GBR[STAMP]2026[STAMP]4821"
    ),
    reasoning="""
        UK visa with entry/exit stamp overlay — common issue with physical travel documents.
        'VISA TYPE: [ENTRY STAMP]USINESS'→'BUSINESS' (B under stamp — 'B' + 'USINESS' = 'BUSINESS').
        'Issue Date: 01-[STAMP]AN-2026'→'01-JAN-2026' (month 3-char code, [STAMP] covers 'J'; 'AN-2026' confirms January).
        'Expiry: 01-[EXIT STAMP]AN-2027'→'01-JAN-2027' (same pattern — one year visa).
        'Surname: [STAMP PARTIAL]ODHARY'→'CHODHARY' or 'CHOUDHARY' — common South Asian surname; 'CHOUDHARY' + 'ODHARY' = first few letters under stamp → likely 'CH' or 'CH' prefix.
        'Given Names: VIKR[EXIT]M SINGH'→'VIKRAM SINGH' — 'A' under stamp in 'VIKRAM' is standard.
        'Nationality: [ENTRY]NDIA'→'INDIA'.
        'Passport No: P[STAMP]4567890'→'P[x]4567890' — one character hidden; Indian passports format is letter+7digits: could be 'PA4567890' etc.
        'Entries Permitted: M[STAMP]LTIPLE'→'MULTIPLE'.
        'Duration of Stay: 9[STAMP] DAYS'→'9[0] DAYS'='90 DAYS' — standard UK business visa duration.
        'Issuing Auth: UKVI'→UK Visas and Immigration — confirmed.
        'Visa No: GBR[STAMP]2026[STAMP]4821'→'GBR[xx]2026[xx]4821' — partially recoverable format.
    """,
    answer="""
        **UK Visa Document (Stamp-Obscured — Reconstructed):**

        | Field              | Reconstructed Value              | Confidence |
        |--------------------|----------------------------------|------------|
        | Visa Type          | BUSINESS                         | High       |
        | Issue Date         | 01-JAN-2026                      | High       |
        | Expiry Date        | 01-JAN-2027 (1-year validity)    | High       |
        | Surname            | [CH/C]HOUDHARY                   | Medium     |
        | Given Names        | VIKRAM SINGH                     | High       |
        | Nationality        | INDIA                            | High       |
        | Passport Number    | P[x]4567890 (one char obscured)  | Medium     |
        | Entries Permitted  | MULTIPLE                         | High       |
        | Duration Per Entry | 90 DAYS                          | High       |
        | Issuing Authority  | UKVI (UK Visas and Immigration)  | High       |
        | Visa Number        | GBR[xx]2026[xx]4821 (partial)   | Low        |

        **Stamp Occlusion Analysis:**
        - Entry stamps: obscure 1–3 characters each (4 instances) — typically over document edges
        - Exit stamps: overlap visa page (2 instances)
        - Pattern: stamps placed per UK Border Force standard practice (corners/edges), but poorly aligned

        **Reconstruction Confidence Notes:**
        - "VIKRAM SINGH" high confidence — 'A' in VIKRAM is the only obscured letter, context unambiguous
        - 90-day duration confirmed as standard UK Standard Visitor/Business Visa allowance
        - Surname prefix unclear — 'CHOUDHARY', 'CHAUDHARY', or 'CHODHARY' all possible

        **Recommendation:** For identity verification, surname must be confirmed against the data page (MRZ line), which should be unaffected by stamps.
    """
))

examples.append(fmt(
    query=(
        "Extract details from this land registry document with an official seal stamp over key fields:\n\n"
        "HM LAND REGISTRY\n"
        "TITLE ABSOLUTE\n"
        "Title Number: [SEAL]N 345[SEAL]78\n"
        "Property: 7 [SEAL PARTIAL]eadow Lane, [SEAL]righton BN[SEAL] 4PQ\n"
        "Tenure: [SEAL]REEHOLD\n"
        "Proprietor: [SEAL]atherine [SEAL]owell\n"
        "Price Paid: £[SEAL]85,000\n"
        "Date Registered: [SEAL]5 February [SEAL]026\n"
        "Lender: [SEAL]ationwide Building [SEAL]ociety\n"
        "Charge Amount: £[SEAL]52,000\n"
        "Title Guarantee: Full"
    ),
    reasoning="""
        HM Land Registry title document — official seal obscures initial characters of many fields.
        Pattern: seal is placed in upper-left of document, radiating outwards — consistently removes first 1-2 chars of each line.
        'Title Number: [SEAL]N 345[SEAL]78'→'[x]N 345[x]78' — UK title numbers format is [county code][number]; 'N' suggests Norfolk or similar; could be 'TN 34578' or 'BN 34578'. Brighton titles often 'ESX' prefix — but 'N' is last visible. Could be 'SN345678' or similar. Flag low confidence.
        '7 [SEAL PARTIAL]eadow Lane'→'7 Meadow Lane' ('M' under seal).
        '[SEAL]righton'→'Brighton'.
        'BN[SEAL] 4PQ'→'BN[x] 4PQ' — Brighton postcodes are BN1-BN45; 'BN1 4PQ' or 'BN2 4PQ' likely.
        '[SEAL]REEHOLD'→'FREEHOLD'.
        '[SEAL]atherine [SEAL]owell'→'Catherine Powell' ('C' under seal both times — common English name).
        '£[SEAL]85,000'→'£[x]85,000' — could be £285,000, £385,000, £485,000; Brighton property range mid-high; £285,000-£485,000 plausible.
        '[SEAL]5 February [SEAL]026'→'[1/2]5 February 2026' — '15' or '25 February 2026'. [SEAL]026→'2026'.
        '[SEAL]ationwide Building [SEAL]ociety'→'Nationwide Building Society'.
        '£[SEAL]52,000'→'£[x]52,000' — likely £152,000 or £252,000 (LTV ratio: if price ~£385,000, charge £252,000 = 65% LTV — reasonable).
    """,
    answer="""
        **HM Land Registry Title Document (Seal-Obscured — Reconstructed):**

        | Field            | Reconstructed Value                     | Confidence |
        |------------------|----------------------------------------|------------|
        | Registry         | HM Land Registry                        | High       |
        | Title Class      | Title Absolute                          | High       |
        | Title Number     | [xx]N 345[x]78 — partial                | Low        |
        | Property         | 7 Meadow Lane, Brighton BN[x] 4PQ      | Medium     |
        | Tenure           | FREEHOLD                                | High       |
        | Proprietor       | Catherine Powell                        | High       |
        | Purchase Price   | £[x]85,000 (£285k–£485k range)         | Medium     |
        | Date Registered  | [1]5 or 25 February 2026               | Medium     |
        | Lender           | Nationwide Building Society             | High       |
        | Charge Amount    | £[x]52,000 (likely £152k–£252k)        | Medium     |
        | Title Guarantee  | Full                                    | High       |

        **Seal Occlusion Pattern:** Official embossed seal positioned top-left, radius ~3cm, removing first 1–2 characters from approximately 9 field values. This is a common filing artifact when seals are applied without margin awareness.

        **Reconstruction Logic:**
        - 'Brighton' confirmed from '[x]righton' — only city beginning with 'B' fitting context
        - 'FREEHOLD'/'NATIONWIDE' reconstructed from single leading letter under seal
        - 'Catherine Powell' high confidence — 'C' initial fits both gaps consistently

        **Financial Note:** Purchase price and charge amount ranges estimated from Brighton property market norms and plausible LTV ratios. For legal use, obtain official copy from Land Registry portal.

        **Confidence:** High on parties, tenure, lender. Low on title number. Medium on price and date.
    """
))

examples.append(fmt(
    query=(
        "Extract company details from this official company registration document with multiple stamp overlays:\n\n"
        "COMPANIES HOUSE\n"
        "CERTIFICATE OF [STAMP1]HANGE OF NAME\n"
        "Previous Name: ALPHA TECH [STAMP1]OLUTIONS LTD\n"
        "New Name: [STAMP2]ETA DIGITAL SERVICES [STAMP1]IMITED\n"
        "Company Number: [STAMP2]2456789\n"
        "Effective Date: [STAMP1]0 April 2026\n"
        "Authorised By: [STAMP2]EGISTRAR OF COMPANIES\n"
        "Signed: [ILLEGIBLE UNDER STAMP2]\n"
        "Filed at: Companies House, Cardiff [STAMP1]F10 3UZ\n"
        "This certificate confirms the company ha[STAMP2] changed its\n"
        "registered name with effect from the date [STAMP1]hown above."
    ),
    reasoning="""
        Companies House name change certificate with two overlapping stamps.
        STAMP1 pattern: removes leading/trailing chars from tokens.
        STAMP2 pattern: overlaps different positions.
        '[STAMP1]HANGE OF NAME'→'CHANGE OF NAME' ('C' under stamp1).
        'ALPHA TECH [STAMP1]OLUTIONS LTD'→'ALPHA TECH SOLUTIONS LTD' ('S' under stamp1).
        '[STAMP2]ETA DIGITAL SERVICES [STAMP1]IMITED'→'BETA DIGITAL SERVICES LIMITED' (B and L under respective stamps).
        '[STAMP2]2456789'→'[x]2456789' — UK company numbers 8 digits; missing first digit. Given 'B' company (Beta), could be '12456789' or '02456789'. Plausible 8-digit number.
        '[STAMP1]0 April 2026'→'[1/2/3]0 April 2026' — likely '10', '20', or '30' April 2026. '30 April 2026' (end of month) is a common effective date for company changes.
        '[STAMP2]EGISTRAR OF COMPANIES'→'REGISTRAR OF COMPANIES'.
        'Companies House, Cardiff [STAMP1]F10 3UZ'→'Cardiff CF10 3UZ' — Companies House actual address is Cardiff CF14 3UZ; 'CF10 3UZ' is also central Cardiff — accept as plausible.
        'ha[STAMP2] changed'→'has changed' ('s' under stamp2).
        '[STAMP1]hown above'→'shown above' ('s' under stamp1).
    """,
    answer="""
        **Companies House Certificate of Name Change (Multi-Stamp Reconstructed):**

        | Field              | Reconstructed Value                   | Confidence |
        |--------------------|---------------------------------------|------------|
        | Document Type      | Certificate of Change of Name         | High       |
        | Previous Name      | ALPHA TECH SOLUTIONS LTD              | High       |
        | New Name           | BETA DIGITAL SERVICES LIMITED         | High       |
        | Company Number     | [x]2456789 (first digit obscured)     | Medium     |
        | Effective Date     | [10/20/30] April 2026                 | Medium     |
        | Authorised By      | Registrar of Companies                | High       |
        | Filed At           | Companies House, Cardiff CF10 3UZ     | High       |
        | Signatory          | Illegible (under STAMP2)              | Low        |

        **Stamp Overlay Analysis:**
        - STAMP1: Filing/received stamp — removes first character of approximately 6 field values
        - STAMP2: Authorisation stamp — obscures first character or short sequences in 4 positions
        - Combined occlusion creates ambiguity at company number (first digit) and effective date (day of month)

        **Verification Note:** Companies House actual address is CF14 3UZ (Companies House Way) — 'CF10 3UZ' may be a mail-handling address. Verify against official filing.

        **Company Number Recovery:** Search Companies House register for 'BETA DIGITAL SERVICES LIMITED' to retrieve full 8-digit number unambiguously.

        **Confidence:** High on company names (sufficient context around each stamp). Medium on effective date and company number. Low on signatory.
    """
))

# ── Write output ──────────────────────────────────────────────────────────────

OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Written {len(examples)} examples to {OUT}")

# Quick validation
with open(OUT, "r", encoding="utf-8") as f:
    lines = f.readlines()

counts = {}
for line in lines:
    obj = json.loads(line)
    cat = obj.get("category", "?")
    counts[cat] = counts.get(cat, 0) + 1
    # verify required keys
    assert "text" in obj and "category" in obj and "difficulty" in obj and "source" in obj
    assert obj["difficulty"] == "hard"
    assert obj["source"] == "claude_quality"
    assert "<|im_start|>system" in obj["text"]
    assert "<think>" in obj["text"]

print(f"Validation passed. Categories: {counts}")
print("Sample keys from first record:", list(json.loads(lines[0]).keys()))
