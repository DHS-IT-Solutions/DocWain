#!/usr/bin/env python3
"""
Generate 30 SFT examples for visualization intelligence training.
Covers 6 scenarios × 5 examples each:
  1. Time series → line chart
  2. Categorical comparison → bar chart
  3. Composition data → pie/donut chart
  4. Distribution data → histogram
  5. Multi-dimensional comparison → heatmap / radar chart
  6. When NOT to visualize → table or text explanation
"""

import json
import os

OUTPUT_PATH = "/home/ubuntu/PycharmProjects/DocWain/finetune_artifacts/sprint/claude_quality/visualization_batch.jsonl"

SYSTEM = (
    "You are DocWain, an enterprise document intelligence assistant. "
    "You analyse documents with deep contextual understanding, extract structured information, "
    "identify patterns and anomalies, and provide holistic analysis grounded in evidence. "
    "You reason step-by-step before answering, state your confidence level, and cite specific sources. "
    "When information is insufficient, you say so clearly rather than guessing."
)

def make_example(query: str, reasoning: str, answer: str, chart_spec: dict | None) -> dict:
    if chart_spec is not None:
        chart_block = "\n\n<chart_spec>\n" + json.dumps(chart_spec, indent=2) + "\n</chart_spec>"
    else:
        chart_block = ""

    text = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n"
        f"{answer}{chart_block}<|im_end|>"
    )
    return {
        "text": text,
        "category": "visualization",
        "difficulty": "hard",
        "source": "claude_quality",
    }


# ─────────────────────────────────────────────
# SCENARIO 1 — Time series → line chart (5)
# ─────────────────────────────────────────────

ts1_query = """Document excerpt — Quarterly Revenue Report FY 2023:

Q1 2023: £2.1 M
Q2 2023: £2.4 M
Q3 2023: £3.0 M
Q4 2023: £3.7 M

How is revenue trending and can you produce a visualisation?"""

ts1_reasoning = """The document provides four sequential quarterly revenue figures across a single fiscal year. This is a classic time-series dataset: the x-axis is ordered, evenly-spaced time intervals (Q1–Q4) and the y-axis is a continuous numerical measure (revenue in £M). Time-series data is best represented as a line chart because:
1. The line visually encodes the direction and rate of change between successive periods.
2. Connecting points highlights the monotone upward trend.
3. A bar chart would also work but de-emphasises the continuity of the trend.

Data points are directly extracted from the excerpt. No imputation required.
Confidence: HIGH — all four data points are explicitly stated."""

ts1_answer = """Revenue shows consistent quarter-on-quarter growth throughout FY 2023, rising from £2.1 M in Q1 to £3.7 M in Q4 — a 76 % increase over the year. The acceleration appears steepest in Q3–Q4, suggesting a strengthening growth trajectory entering the new fiscal year.

A line chart is the most appropriate visualisation to communicate this upward trend clearly."""

ts1_spec = {
    "type": "line",
    "title": "Quarterly Revenue FY 2023",
    "x_axis": "Quarter",
    "y_axis": "Revenue (£M)",
    "data": [
        {"x": "Q1 2023", "y": 2.1},
        {"x": "Q2 2023", "y": 2.4},
        {"x": "Q3 2023", "y": 3.0},
        {"x": "Q4 2023", "y": 3.7},
    ],
    "insights": "Monotone upward trend; 76% growth Q1→Q4; steepest gain in Q3–Q4 segment.",
}

# ---

ts2_query = """Monthly Active Users report (Jan–Jun 2024):

January: 14 200
February: 15 800
March: 18 400
April: 17 900
May: 20 100
June: 22 500

Analyse the user growth trend and suggest a chart."""

ts2_reasoning = """Six consecutive monthly observations of a continuous metric (Monthly Active Users) constitute a time series. The x-axis is calendar month (ordinal, evenly spaced), and y-axis is the user count. A line chart is ideal because:
1. It preserves temporal ordering and shows momentum.
2. The slight dip in April is clearly visible as a local trough on a line — a bar chart would obscure this comparative signal.
3. With only 6 points, a line chart avoids visual clutter.

Confidence: HIGH — all six values are explicitly stated."""

ts2_answer = """MAU grew from 14 200 in January to 22 500 in June (+58 %). A brief dip in April (−2.7 % MoM) interrupts the otherwise consistent growth trajectory, possibly reflecting seasonal behaviour or a temporary product issue. Recovery and acceleration in May–June suggest the dip was transient.

A line chart best captures this trend with the April anomaly clearly visible."""

ts2_spec = {
    "type": "line",
    "title": "Monthly Active Users — H1 2024",
    "x_axis": "Month",
    "y_axis": "Monthly Active Users",
    "data": [
        {"x": "Jan", "y": 14200},
        {"x": "Feb", "y": 15800},
        {"x": "Mar", "y": 18400},
        {"x": "Apr", "y": 17900},
        {"x": "May", "y": 20100},
        {"x": "Jun", "y": 22500},
    ],
    "insights": "Overall +58% H1 growth; April dip of -2.7% MoM is a notable anomaly; strong recovery May–Jun.",
}

# ---

ts3_query = """Energy consumption log — Office Building A, 2023 (MWh per month):

Jan 42, Feb 38, Mar 31, Apr 25, May 18, Jun 15, Jul 14, Aug 16, Sep 21, Oct 28, Nov 35, Dec 43

Visualise energy usage patterns and explain what the chart shows."""

ts3_reasoning = """Twelve monthly energy readings form a complete annual time series. The data has a clear seasonal pattern (high in winter, low in summer) which is best surfaced by a line chart because:
1. Connecting 12 points in calendar order reveals the U-shaped seasonal cycle visually.
2. The line makes it easy to see the nadir (July: 14 MWh) and twin peaks (Jan: 42, Dec: 43 MWh).
3. A histogram would lose temporal ordering; a bar chart could work but doesn't emphasise the cyclic continuity.

Confidence: HIGH — all 12 values are explicitly provided."""

ts3_answer = """Energy consumption follows a clear inverse-seasonal pattern, peaking in winter (Jan 42 MWh, Dec 43 MWh) and reaching its lowest point in July (14 MWh). This is consistent with heating demand dominating winter usage. The pattern is symmetric around the mid-year trough, suggesting the building's energy consumption is primarily HVAC-driven.

A line chart over 12 months best communicates this seasonal cycle."""

ts3_spec = {
    "type": "line",
    "title": "Monthly Energy Consumption — Office Building A, 2023",
    "x_axis": "Month",
    "y_axis": "Energy Consumption (MWh)",
    "data": [
        {"x": "Jan", "y": 42}, {"x": "Feb", "y": 38}, {"x": "Mar", "y": 31},
        {"x": "Apr", "y": 25}, {"x": "May", "y": 18}, {"x": "Jun", "y": 15},
        {"x": "Jul", "y": 14}, {"x": "Aug", "y": 16}, {"x": "Sep", "y": 21},
        {"x": "Oct", "y": 28}, {"x": "Nov", "y": 35}, {"x": "Dec", "y": 43},
    ],
    "insights": "Inverse-seasonal pattern; winter peak ~42-43 MWh; summer trough 14 MWh (July); U-shaped annual cycle.",
}

# ---

ts4_query = """Clinical trial document — Patient recruitment weekly progress:

Week 1: 8 patients
Week 2: 14 patients
Week 3: 19 patients
Week 4: 22 patients
Week 5: 21 patients
Week 6: 27 patients
Week 7: 30 patients
Week 8: 28 patients

Trial target: 200 patients. What is the recruitment trend?"""

ts4_reasoning = """Eight consecutive weekly recruitment counts form a time series with moderate volatility. A line chart is appropriate because:
1. Weekly measurements are sequential and evenly spaced — line encoding captures the trend and week-to-week variation.
2. Minor dips (Week 5: 21, Week 8: 28) are visible as local troughs, revealing recruitment pace inconsistencies.
3. Cumulative totals could supplement but the question asks about trend, which a line chart addresses directly.

Confidence: HIGH — eight explicit values provided. Cumulative total after 8 weeks: 169 patients."""

ts4_answer = """Recruitment shows a generally upward trend, rising from 8 patients/week to 28–30 by Weeks 7–8. Slight dips in Weeks 5 and 8 suggest minor variability, possibly reflecting site capacity or holiday periods. Cumulative enrolment through Week 8 is 169 patients (85 % of the 200 target), indicating the trial is on track to meet its target within the next 1–2 weeks if current pace is maintained.

A line chart over the 8-week period communicates both the trend and the week-level variation."""

ts4_spec = {
    "type": "line",
    "title": "Weekly Patient Recruitment — Clinical Trial",
    "x_axis": "Week",
    "y_axis": "Patients Recruited",
    "data": [
        {"x": "Week 1", "y": 8}, {"x": "Week 2", "y": 14}, {"x": "Week 3", "y": 19},
        {"x": "Week 4", "y": 22}, {"x": "Week 5", "y": 21}, {"x": "Week 6", "y": 27},
        {"x": "Week 7", "y": 30}, {"x": "Week 8", "y": 28},
    ],
    "insights": "Cumulative 169/200 patients (85% target); generally upward trend; minor dips Week 5 & 8.",
}

# ---

ts5_query = """Supply chain report — Lead time (days) from supplier over 10 months:

Month 1: 12, Month 2: 11, Month 3: 13, Month 4: 15, Month 5: 18, Month 6: 22, Month 7: 25, Month 8: 23, Month 9: 20, Month 10: 17

Is supply chain performance improving or deteriorating?"""

ts5_reasoning = """Ten monthly lead-time readings form a time series. This is the most suitable scenario for a line chart because:
1. The metric is continuous and measured at regular monthly intervals.
2. The line will immediately show the "rise then fall" pattern — a spike in Months 5–7 followed by recovery.
3. Trend direction is the key question, and a line chart answers it visually at a glance.

The data shows a clear inverted-V shape, peaking at Month 7 (25 days).
Confidence: HIGH — all 10 values explicit."""

ts5_answer = """Lead times deteriorated significantly from Month 1 (12 days) through Month 7 (25 days) — a 108 % increase — before recovering to 17 days by Month 10. The pattern suggests a supply disruption peaking around Months 6–7 that is now abating. Performance has not yet returned to baseline levels (12–13 days), so recovery is partial. Continued monitoring is recommended.

A line chart clearly communicates the disruption spike and subsequent recovery trajectory."""

ts5_spec = {
    "type": "line",
    "title": "Monthly Supplier Lead Time — 10-Month Trend",
    "x_axis": "Month",
    "y_axis": "Lead Time (Days)",
    "data": [
        {"x": "Month 1", "y": 12}, {"x": "Month 2", "y": 11}, {"x": "Month 3", "y": 13},
        {"x": "Month 4", "y": 15}, {"x": "Month 5", "y": 18}, {"x": "Month 6", "y": 22},
        {"x": "Month 7", "y": 25}, {"x": "Month 8", "y": 23}, {"x": "Month 9", "y": 20},
        {"x": "Month 10", "y": 17},
    ],
    "insights": "Peak disruption Month 7 (25 days, +108% vs baseline); recovery underway but not complete by Month 10.",
}

# ─────────────────────────────────────────────
# SCENARIO 2 — Categorical comparison → bar chart (5)
# ─────────────────────────────────────────────

bar1_query = """Regional sales performance report — FY 2024:

North: £4.2 M
South: £3.8 M
East: £5.1 M
West: £2.9 M
Central: £4.7 M

Which regions are outperforming and which need attention?"""

bar1_reasoning = """Five discrete, unordered categories (regions) each with a single revenue value — this is a categorical comparison dataset. A bar chart is optimal because:
1. Each bar represents one category; bar length encodes magnitude, enabling direct comparison.
2. Categories have no inherent ordering (unlike time), so a line chart's implied continuity would be misleading.
3. Sorting bars by value (descending) makes rank order immediately clear.

Confidence: HIGH — five explicit values provided."""

bar1_answer = """East region leads with £5.1 M, followed by Central (£4.7 M) and North (£4.2 M). South (£3.8 M) is below the £4.14 M company average, and West (£2.9 M) is the clear underperformer at 30 % below average. East and Central together account for 48 % of total revenue (£20.7 M). West warrants a targeted performance review.

A sorted bar chart provides the clearest regional comparison."""

bar1_spec = {
    "type": "bar",
    "title": "Regional Sales Performance — FY 2024",
    "x_axis": "Region",
    "y_axis": "Revenue (£M)",
    "data": [
        {"category": "East", "value": 5.1},
        {"category": "Central", "value": 4.7},
        {"category": "North", "value": 4.2},
        {"category": "South", "value": 3.8},
        {"category": "West", "value": 2.9},
    ],
    "insights": "East leads (+23% above avg); West underperforms (-30% below avg); top-2 regions = 48% of total revenue.",
}

# ---

bar2_query = """HR report — Average employee satisfaction scores by department (scale 1–10):

Engineering: 7.8
Sales: 6.2
Operations: 7.1
Finance: 8.3
HR: 7.5
Marketing: 6.9

Which departments have the lowest satisfaction and need priority attention?"""

bar2_reasoning = """Six unordered departments with a single continuous score each — classic categorical comparison. Bar chart is appropriate because:
1. Departments are nominal categories; bars prevent false impression of continuity between them.
2. A horizontal bar chart is especially clear when category labels are long words.
3. A reference line at company average (7.3) could be overlaid to highlight below-average departments.

Average = (7.8+6.2+7.1+8.3+7.5+6.9)/6 = 7.3.
Confidence: HIGH — all values explicit."""

bar2_answer = """Finance scores highest (8.3/10) and Engineering also performs well (7.8). Sales (6.2) and Marketing (6.9) fall below the company average of 7.3, with Sales showing the largest gap (−1.1 points). HR interventions should prioritise Sales and Marketing departments. Operations (7.1) is marginally below average and warrants monitoring.

A bar chart sorted by score, with a reference line at 7.3, communicates the ranking and deviation from average most clearly."""

bar2_spec = {
    "type": "bar",
    "title": "Employee Satisfaction by Department (1–10 Scale)",
    "x_axis": "Department",
    "y_axis": "Satisfaction Score",
    "data": [
        {"category": "Finance", "value": 8.3},
        {"category": "Engineering", "value": 7.8},
        {"category": "HR", "value": 7.5},
        {"category": "Operations", "value": 7.1},
        {"category": "Marketing", "value": 6.9},
        {"category": "Sales", "value": 6.2},
    ],
    "insights": "Company average 7.3; Sales (-1.1) and Marketing (-0.4) below average; Finance highest at 8.3.",
}

# ---

bar3_query = """Procurement audit — Vendor on-time delivery rate (%) for Q3 2024:

Vendor A: 94%
Vendor B: 87%
Vendor C: 78%
Vendor D: 95%
Vendor E: 82%
Vendor F: 91%

Identify vendor performance gaps and recommend visualisation."""

bar3_reasoning = """Six discrete vendors with a single percentage metric — categorical comparison. Bar chart is ideal:
1. Each vendor is a distinct entity with no ordinal relationship — bars avoid implying a trend.
2. Sorting descending allows immediate identification of the best and worst performers.
3. A reference line at a contractual SLA (e.g., 90%) would highlight compliance breaches.

Confidence: HIGH — all six percentages explicit."""

bar3_answer = """Vendor D leads at 95 % on-time delivery, followed closely by Vendor A (94 %). Vendors B (87 %), E (82 %), and C (78 %) fall below the typical 90 % SLA threshold. Vendor C is the critical concern at 78 % — 12 points below the SLA. Vendors B and E represent moderate non-compliance. Procurement should initiate formal performance improvement conversations with Vendors C, E, and B in order of priority.

A bar chart with an SLA reference line at 90 % provides the clearest compliance gap view."""

bar3_spec = {
    "type": "bar",
    "title": "Vendor On-Time Delivery Rate — Q3 2024",
    "x_axis": "Vendor",
    "y_axis": "On-Time Delivery Rate (%)",
    "data": [
        {"category": "Vendor D", "value": 95},
        {"category": "Vendor A", "value": 94},
        {"category": "Vendor F", "value": 91},
        {"category": "Vendor B", "value": 87},
        {"category": "Vendor E", "value": 82},
        {"category": "Vendor C", "value": 78},
    ],
    "insights": "SLA threshold 90%; Vendors C, E, B non-compliant; Vendor C worst at 78% (-12pp below SLA).",
}

# ---

bar4_query = """Academic report — Exam pass rates by subject, Year 10 cohort:

Mathematics: 72%
English: 85%
Science: 68%
History: 79%
Geography: 81%
Art: 93%

Which subjects have the weakest pass rates?"""

bar4_reasoning = """Six academic subjects (nominal categories) with a single pass-rate percentage each. Bar chart is the right choice:
1. Subjects are unordered categories; a bar chart avoids implying a time trend or ordered relationship.
2. Sorting descending makes rank immediately readable.
3. A school benchmark line (e.g., 80%) would clearly show which subjects are below target.

Mean pass rate = (72+85+68+79+81+93)/6 ≈ 79.7%.
Confidence: HIGH."""

bar4_answer = """Art (93 %) and English (85 %) are strongest. Geography (81 %) and History (79 %) are near the cohort average (79.7 %). Mathematics (72 %) and Science (68 %) are below average and below a typical 80 % benchmark, with Science being the most concerning. Targeted intervention programmes for Science and Mathematics are recommended.

A descending bar chart with an 80 % benchmark line communicates the gap most effectively."""

bar4_spec = {
    "type": "bar",
    "title": "Year 10 Exam Pass Rates by Subject",
    "x_axis": "Subject",
    "y_axis": "Pass Rate (%)",
    "data": [
        {"category": "Art", "value": 93},
        {"category": "English", "value": 85},
        {"category": "Geography", "value": 81},
        {"category": "History", "value": 79},
        {"category": "Mathematics", "value": 72},
        {"category": "Science", "value": 68},
    ],
    "insights": "Science (68%) and Maths (72%) below 80% benchmark; Art leads at 93%; cohort avg 79.7%.",
}

# ---

bar5_query = """IT helpdesk report — Ticket volume by category, October 2024:

Password Reset: 310
Software Installation: 185
Hardware Fault: 97
Network Issue: 142
Access Request: 228
Other: 54

Where should helpdesk resource be focused?"""

bar5_reasoning = """Six helpdesk ticket categories (nominal) with associated counts — categorical comparison. Bar chart is appropriate:
1. Categories are unordered, discrete items; bars make magnitude comparison straightforward.
2. Sorting by volume descending will highlight the highest-burden categories immediately.
3. Total tickets = 1016; percentages can supplement to show relative burden.

Confidence: HIGH — all category counts explicit."""

bar5_answer = """Password Reset dominates at 310 tickets (30.5 % of total), followed by Access Request (228, 22.4 %) and Software Installation (185, 18.2 %). These top three categories account for 71 % of all tickets. Automation investments — self-service password reset and access provisioning workflows — would deliver the greatest reduction in helpdesk load. Hardware Fault (97) and Other (54) are the lowest-volume categories.

A descending bar chart clearly shows where resource concentration will have the most impact."""

bar5_spec = {
    "type": "bar",
    "title": "Helpdesk Ticket Volume by Category — October 2024",
    "x_axis": "Category",
    "y_axis": "Ticket Count",
    "data": [
        {"category": "Password Reset", "value": 310},
        {"category": "Access Request", "value": 228},
        {"category": "Software Installation", "value": 185},
        {"category": "Network Issue", "value": 142},
        {"category": "Hardware Fault", "value": 97},
        {"category": "Other", "value": 54},
    ],
    "insights": "Top 3 categories = 71% of volume; password reset alone = 30.5%; automation opportunity high.",
}

# ─────────────────────────────────────────────
# SCENARIO 3 — Composition data → pie/donut chart (5)
# ─────────────────────────────────────────────

pie1_query = """Annual budget allocation document — FY 2025 total budget: £10 M

Personnel: £4.8 M
Technology: £2.1 M
Marketing: £1.5 M
Operations: £0.9 M
R&D: £0.7 M

Visualise how the budget is composed."""

pie1_reasoning = """Five budget categories that sum to a whole (£10 M total, the document's stated total). This is part-to-whole composition data, which is best represented by a pie or donut chart because:
1. The key question is "what share of the total does each category hold?" — this is precisely what pie charts answer.
2. With only 5 segments, readability is maintained (pie charts become unwieldy beyond 6-7 segments).
3. A stacked bar chart is an alternative for a single entity's composition, but pie charts are more intuitive for this use case.

Note: 4.8+2.1+1.5+0.9+0.7 = 10.0 — confirms completeness.
Confidence: HIGH."""

pie1_answer = """Personnel dominates the budget at 48 %, nearly half of total spend — typical for a service-oriented organisation. Technology (21 %) and Marketing (15 %) are the next largest allocations. Operations (9 %) and R&D (7 %) are relatively modest; the low R&D share may warrant strategic review if innovation is a growth driver.

A donut chart communicates the composition at a glance, with each segment proportional to its budget share."""

pie1_spec = {
    "type": "pie",
    "title": "FY 2025 Budget Allocation (Total: £10 M)",
    "x_axis": None,
    "y_axis": "Budget Share (%)",
    "data": [
        {"segment": "Personnel", "value": 4.8, "percentage": 48.0},
        {"segment": "Technology", "value": 2.1, "percentage": 21.0},
        {"segment": "Marketing", "value": 1.5, "percentage": 15.0},
        {"segment": "Operations", "value": 0.9, "percentage": 9.0},
        {"segment": "R&D", "value": 0.7, "percentage": 7.0},
    ],
    "insights": "Personnel = 48% dominance; top 3 categories = 84% of total; R&D underfunded relative to growth ambitions.",
}

# ---

pie2_query = """Customer demographics report — Age group breakdown, platform users (n=25 000):

18–24: 4 750
25–34: 8 250
35–44: 6 500
45–54: 3 500
55+: 2 000

Show the age composition of the user base."""

pie2_reasoning = """Five age segments summing to 25 000 (the stated total). This is a part-to-whole composition question. A donut/pie chart is appropriate because:
1. Each age group is a portion of the whole user base; the question asks for a composition view.
2. Five segments are within the readable limit for pie charts.
3. Percentages can be labelled directly on segments for clarity.

Check: 4750+8250+6500+3500+2000 = 25 000. Confirmed.
Confidence: HIGH."""

pie2_answer = """The 25–34 age group is the largest segment at 33 %, making it the core demographic. Combined with 35–44 (26 %), the 25–44 bracket accounts for 59 % of the user base — a predominantly millennial/Gen-X audience. The 18–24 segment (19 %) shows strong younger adoption. Users 55+ represent only 8 %, suggesting an opportunity to grow adoption among older demographics.

A donut chart shows these proportions clearly, with each segment labelled by percentage."""

pie2_spec = {
    "type": "pie",
    "title": "User Age Distribution (n = 25,000)",
    "x_axis": None,
    "y_axis": "Share of Users (%)",
    "data": [
        {"segment": "18–24", "value": 4750, "percentage": 19.0},
        {"segment": "25–34", "value": 8250, "percentage": 33.0},
        {"segment": "35–44", "value": 6500, "percentage": 26.0},
        {"segment": "45–54", "value": 3500, "percentage": 14.0},
        {"segment": "55+", "value": 2000, "percentage": 8.0},
    ],
    "insights": "25-34 is core segment (33%); 25-44 combined = 59%; 55+ underrepresented at 8%.",
}

# ---

pie3_query = """Energy mix report — Electricity generation sources, National Grid Q2 2024:

Natural Gas: 38%
Wind: 28%
Nuclear: 19%
Solar: 9%
Other Renewables: 6%

Analyse the composition of the energy mix."""

pie3_reasoning = """Five generation source categories with percentage shares that sum to 100%. This is a canonical composition dataset for a pie/donut chart:
1. Each source is a fraction of the whole generation mix.
2. Five slices maintain readability.
3. The shares are already given as percentages, confirming part-to-whole structure.

Confidence: HIGH — percentages are explicit and sum to 100%."""

pie3_answer = """Natural Gas remains the largest single source at 38 %, but renewables (Wind 28 % + Solar 9 % + Other 6 %) collectively account for 43 % of generation — exceeding fossil fuels for the first time if this trend continues. Nuclear (19 %) provides stable baseload. The direction of travel towards renewables is clear; however, gas dependency still poses carbon and price-volatility risks.

A donut chart communicates the relative scale of each source and the fossil vs. renewable balance at a glance."""

pie3_spec = {
    "type": "pie",
    "title": "Electricity Generation Mix — Q2 2024",
    "x_axis": None,
    "y_axis": "Share of Generation (%)",
    "data": [
        {"segment": "Natural Gas", "value": 38, "percentage": 38},
        {"segment": "Wind", "value": 28, "percentage": 28},
        {"segment": "Nuclear", "value": 19, "percentage": 19},
        {"segment": "Solar", "value": 9, "percentage": 9},
        {"segment": "Other Renewables", "value": 6, "percentage": 6},
    ],
    "insights": "Renewables collectively 43% vs Gas 38%; nuclear provides stable 19% baseload; clean energy majority emerging.",
}

# ---

pie4_query = """Website traffic sources — Analytics report, March 2024:

Organic Search: 42%
Direct: 21%
Paid Search: 16%
Social Media: 12%
Referral: 6%
Email: 3%

What does the traffic composition tell us about acquisition channels?"""

pie4_reasoning = """Six traffic source categories summing to 100% — a part-to-whole composition view. Pie/donut chart is suitable:
1. Each source is a distinct slice of total traffic; the question asks about composition and relative importance.
2. Six segments is near the upper limit for pie readability; a donut with labelled segments manages this.
3. A bar chart could also work to show absolute differences but would not convey "share of whole" as intuitively.

Confidence: HIGH — all percentages explicit and sum to 100%."""

pie4_answer = """Organic search dominates at 42 %, indicating strong SEO performance. Direct traffic (21 %) shows brand recognition. Paid search (16 %) contributes meaningfully but at a cost. Social (12 %) and referral (6 %) are secondary channels. Email (3 %) is underperforming relative to its typical ROI potential — a newsletter or lifecycle email programme could increase this share significantly.

A donut chart with six labelled segments illustrates the channel mix and highlights organic's dominant role."""

pie4_spec = {
    "type": "pie",
    "title": "Website Traffic Sources — March 2024",
    "x_axis": None,
    "y_axis": "Share of Sessions (%)",
    "data": [
        {"segment": "Organic Search", "value": 42, "percentage": 42},
        {"segment": "Direct", "value": 21, "percentage": 21},
        {"segment": "Paid Search", "value": 16, "percentage": 16},
        {"segment": "Social Media", "value": 12, "percentage": 12},
        {"segment": "Referral", "value": 6, "percentage": 6},
        {"segment": "Email", "value": 3, "percentage": 3},
    ],
    "insights": "Organic dominates at 42%; email at 3% is underutilised; paid+organic = 58% of all traffic.",
}

# ---

pie5_query = """Project resource allocation — software development project:

Development: 45%
Testing & QA: 20%
Project Management: 15%
Design & UX: 12%
DevOps & Infrastructure: 8%

Is this resource allocation appropriate for an enterprise software project?"""

pie5_reasoning = """Five resource allocation categories summing to 100% — a composition dataset. Pie chart is appropriate because:
1. The question asks about how the whole project budget/effort is divided — classic part-to-whole framing.
2. Five segments are clearly readable.
3. Industry benchmarks (e.g., testing should typically be 20–30%) can be referenced alongside the chart.

Confidence: HIGH — percentages explicit, sum to 100%."""

pie5_answer = """Development at 45% is within the expected range for enterprise software. Testing & QA at 20% is adequate but towards the lower end — for high-criticality enterprise software, 25–30% is recommended to reduce post-release defects. Project Management (15%) appears slightly high; above 10% often indicates process overhead. Design & UX at 12% reflects a reasonable user-centred investment. DevOps at 8% may be insufficient for a cloud-native or CI/CD-heavy project.

A donut chart visualises these proportions clearly and allows comparison against industry benchmarks."""

pie5_spec = {
    "type": "pie",
    "title": "Project Resource Allocation — Software Development",
    "x_axis": None,
    "y_axis": "Resource Share (%)",
    "data": [
        {"segment": "Development", "value": 45, "percentage": 45},
        {"segment": "Testing & QA", "value": 20, "percentage": 20},
        {"segment": "Project Management", "value": 15, "percentage": 15},
        {"segment": "Design & UX", "value": 12, "percentage": 12},
        {"segment": "DevOps & Infrastructure", "value": 8, "percentage": 8},
    ],
    "insights": "Testing below recommended 25-30%; PM overhead at 15%; DevOps may need uplifting for cloud-native delivery.",
}

# ─────────────────────────────────────────────
# SCENARIO 4 — Distribution data → histogram (5)
# ─────────────────────────────────────────────

hist1_query = """HR analytics — Time-to-hire (days) for 50 new hires, Q1 2024:

Bin 0–14 days: 6 hires
Bin 15–29 days: 12 hires
Bin 30–44 days: 18 hires
Bin 45–59 days: 9 hires
Bin 60–74 days: 4 hires
Bin 75+ days: 1 hire

Analyse the distribution of time-to-hire and suggest a visualisation."""

hist1_reasoning = """The data is already pre-binned into equal-width ranges (15-day bins) with frequency counts — this is explicitly a distribution dataset. A histogram is the canonical choice because:
1. It shows the shape of the distribution (here: roughly normal/slightly right-skewed).
2. The mode bin (30–44 days, 18 hires) is immediately visible as the tallest bar.
3. Outliers (75+ days) are visible in the long tail.
4. A bar chart looks similar but histogram is technically correct for continuous binned data (bars touch; no gaps).

Total = 6+12+18+9+4+1 = 50. Confirmed.
Confidence: HIGH."""

hist1_answer = """Time-to-hire follows a roughly normal distribution with a slight right skew. The mode is 30–44 days (36 % of hires), and the median likely falls within this bin. 36 % of hires are completed within 29 days, while 10 % take 60+ days — the long tail represents potential process bottlenecks or hard-to-fill roles. A target of sub-45 days would capture 72 % of current hires with modest process improvement.

A histogram communicates the distribution shape, central tendency, and tail characteristics most effectively."""

hist1_spec = {
    "type": "histogram",
    "title": "Time-to-Hire Distribution — Q1 2024 (n=50)",
    "x_axis": "Time-to-Hire (Days)",
    "y_axis": "Number of Hires",
    "data": [
        {"bin": "0–14", "frequency": 6},
        {"bin": "15–29", "frequency": 12},
        {"bin": "30–44", "frequency": 18},
        {"bin": "45–59", "frequency": 9},
        {"bin": "60–74", "frequency": 4},
        {"bin": "75+", "frequency": 1},
    ],
    "insights": "Mode bin: 30-44 days; right-skewed; 10% of hires take 60+ days indicating tail bottleneck.",
}

# ---

hist2_query = """Customer service report — Call handle time distribution, 200 calls sampled:

Under 2 min: 14 calls
2–4 min: 38 calls
4–6 min: 62 calls
6–8 min: 48 calls
8–10 min: 24 calls
Over 10 min: 14 calls

What does the distribution of call handle times show?"""

hist2_reasoning = """Pre-binned frequency data for a continuous variable (call duration). Histogram is the correct visualisation because:
1. Call duration is continuous, and bins are equal-width (2-minute intervals).
2. The distribution shape (appears normal, centred around 4–6 min) is the key insight.
3. A histogram reveals whether handle times cluster tightly (consistent agent performance) or are widely spread (variable quality).

Total = 14+38+62+48+24+14 = 200. Confirmed.
Confidence: HIGH."""

hist2_answer = """Call handle times follow an approximately normal distribution, centred on the 4–6 minute bin (31 % of calls). The distribution is relatively symmetric, with similar volumes at the tails (under 2 min: 7 %, over 10 min: 7 %). This suggests consistent agent performance with the bulk of calls handled in 4–8 minutes. The 7 % of calls exceeding 10 minutes should be reviewed for complexity patterns or training opportunities.

A histogram with equal 2-minute bins communicates the distributional shape clearly."""

hist2_spec = {
    "type": "histogram",
    "title": "Call Handle Time Distribution (n=200 calls)",
    "x_axis": "Handle Time (Minutes)",
    "y_axis": "Number of Calls",
    "data": [
        {"bin": "<2", "frequency": 14},
        {"bin": "2–4", "frequency": 38},
        {"bin": "4–6", "frequency": 62},
        {"bin": "6–8", "frequency": 48},
        {"bin": "8–10", "frequency": 24},
        {"bin": ">10", "frequency": 14},
    ],
    "insights": "Near-normal distribution; mode 4-6 min; symmetric tails; 7% outlier calls >10 min warrant review.",
}

# ---

hist3_query = """Finance report — Invoice payment delays (days past due) for 120 invoices:

0–7 days late: 45 invoices
8–14 days late: 30 invoices
15–21 days late: 22 invoices
22–28 days late: 14 invoices
29–35 days late: 6 invoices
36+ days late: 3 invoices

Analyse the payment delay distribution."""

hist3_reasoning = """Six bins of equal width (7 days) with frequency counts — classic binned distribution data. Histogram is appropriate:
1. Payment delay is a continuous variable; bins are equal-width; histogram bars should be contiguous (no gaps).
2. The distribution is clearly right-skewed (exponential decay) — most late invoices are only slightly late; few are severely overdue.
3. The shape informs collections strategy: the bulk of overdue value is recoverable quickly.

Total = 45+30+22+14+6+3 = 120. Confirmed.
Confidence: HIGH."""

hist3_answer = """Payment delays are strongly right-skewed: 37.5 % of late invoices are recovered within 7 days, and 62.5 % within 14 days. However, 7.5 % of invoices (9 out of 120) are more than 28 days overdue — these represent the highest credit risk and should be escalated to formal collections. The skewed distribution is typical of trade receivables and suggests most customers pay with minimal prompting.

A histogram of 7-day bins visually communicates this skewed pattern and the long-overdue tail."""

hist3_spec = {
    "type": "histogram",
    "title": "Invoice Payment Delay Distribution (n=120)",
    "x_axis": "Days Past Due",
    "y_axis": "Number of Invoices",
    "data": [
        {"bin": "0–7", "frequency": 45},
        {"bin": "8–14", "frequency": 30},
        {"bin": "15–21", "frequency": 22},
        {"bin": "22–28", "frequency": 14},
        {"bin": "29–35", "frequency": 6},
        {"bin": "36+", "frequency": 3},
    ],
    "insights": "Right-skewed; 62.5% recover within 14 days; 7.5% >28 days overdue require escalation.",
}

# ---

hist4_query = """Product quality report — Defect rate (%) per production batch, 80 batches audited:

0.0–0.5%: 8 batches
0.5–1.0%: 22 batches
1.0–1.5%: 28 batches
1.5–2.0%: 14 batches
2.0–2.5%: 6 batches
2.5%+: 2 batches

Is production quality within acceptable limits?"""

hist4_reasoning = """80 production batches binned by defect rate — this is a distribution of a quality metric. Histogram is the right visualisation because:
1. Defect rate is continuous; the bins are equal-width (0.5 pp).
2. The shape (slightly right-skewed, mode 1.0–1.5%) reveals the central tendency and how many batches breach any SLA threshold.
3. If the acceptable defect threshold is, say, 2%, we can immediately calculate the compliance rate from the histogram.

Total = 8+22+28+14+6+2 = 80. Confirmed.
Confidence: HIGH."""

hist4_answer = """Defect rates follow a roughly normal distribution with a slight right skew, centred on the 1.0–1.5 % bin (35 % of batches). 73.75 % of batches fall below 1.5 % defect rate. However, 10 % of batches (8 out of 80) exceed 2 %, which is typically considered the upper control limit for high-precision manufacturing. The two batches at 2.5 %+ require immediate root cause investigation.

A histogram communicates the distribution shape, mode, and out-of-spec proportion clearly."""

hist4_spec = {
    "type": "histogram",
    "title": "Production Batch Defect Rate Distribution (n=80)",
    "x_axis": "Defect Rate (%)",
    "y_axis": "Number of Batches",
    "data": [
        {"bin": "0.0–0.5%", "frequency": 8},
        {"bin": "0.5–1.0%", "frequency": 22},
        {"bin": "1.0–1.5%", "frequency": 28},
        {"bin": "1.5–2.0%", "frequency": 14},
        {"bin": "2.0–2.5%", "frequency": 6},
        {"bin": "2.5%+", "frequency": 2},
    ],
    "insights": "Mode 1.0-1.5%; 10% of batches exceed 2% UCL; 2 batches at critical 2.5%+ threshold.",
}

# ---

hist5_query = """E-commerce report — Order value distribution, 500 orders placed in November 2024:

£0–£25: 62 orders
£25–£50: 118 orders
£50–£100: 172 orders
£100–£200: 105 orders
£200–£500: 36 orders
£500+: 7 orders

What does the order value distribution tell us about customer purchasing behaviour?"""

hist5_reasoning = """Six pre-binned order value ranges with frequency counts — distribution data. Histogram is appropriate:
1. Order value is a continuous variable with natural right-skew (few very high orders, many modest ones).
2. The histogram will reveal the mode range, the shape (right-skewed for revenue), and the high-value tail.
3. Variable bin widths here (£0-25=25, £25-50=25, £50-100=50, £100-200=100, £200-500=300, £500+=open) — worth noting this in the insight. Frequency density would be more precise but frequency count is as given.

Total = 62+118+172+105+36+7 = 500. Confirmed.
Confidence: HIGH."""

hist5_answer = """Order values are right-skewed with the modal range at £50–£100 (34.4 % of orders). Combined, £25–£100 accounts for 58 % of all orders, anchoring the "typical" customer purchase around the £50–£75 range. The high-value tail (£200+) represents only 43 orders (8.6 %) but likely contributes disproportionately to revenue. Targeted upsell strategies for the £50–£100 cohort could shift the distribution rightward. The 62 low-value orders (£0–£25) may reflect first-time buyers or trial purchases.

A histogram reveals the right-skewed nature of order values and identifies the dominant spending band."""

hist5_spec = {
    "type": "histogram",
    "title": "Order Value Distribution — November 2024 (n=500)",
    "x_axis": "Order Value (£)",
    "y_axis": "Number of Orders",
    "data": [
        {"bin": "£0–£25", "frequency": 62},
        {"bin": "£25–£50", "frequency": 118},
        {"bin": "£50–£100", "frequency": 172},
        {"bin": "£100–£200", "frequency": 105},
        {"bin": "£200–£500", "frequency": 36},
        {"bin": "£500+", "frequency": 7},
    ],
    "insights": "Right-skewed; modal band £50-100 (34.4%); high-value tail (£200+) = 8.6% orders; upsell opportunity in modal cohort.",
}

# ─────────────────────────────────────────────
# SCENARIO 5 — Multi-dimensional → heatmap / radar chart (5)
# ─────────────────────────────────────────────

heatmap1_query = """Customer support performance matrix — Response time SLA compliance (%) by team and day of week:

           Mon   Tue   Wed   Thu   Fri
Team A:    94    96    92    89    78
Team B:    88    91    87    85    72
Team C:    97    98    96    94    88
Team D:    82    84    80    76    61

Analyse SLA compliance patterns across teams and days."""

heatmap1_reasoning = """A 4×5 matrix of SLA compliance percentages with two categorical dimensions (team × day of week). A heatmap is the ideal choice because:
1. It encodes the third dimension (compliance %) as colour intensity across a grid.
2. Patterns along rows (team differences) and columns (weekday patterns) are simultaneously visible.
3. With 20 cells, a table is harder to scan for patterns than a colour-coded grid.

Key patterns visible: all teams decline on Friday; Team C is consistently best; Team D is consistently weakest.
Confidence: HIGH — all 20 values explicit."""

heatmap1_answer = """The heatmap reveals two clear patterns:
1. **Day-of-week degradation**: All teams show declining SLA compliance towards Friday. Friday scores are 10–21 points lower than Monday scores — suggesting end-of-week resource or capacity issues.
2. **Team hierarchy**: Team C consistently leads (88–98%); Team D consistently underperforms (61–84%), with Friday performance at 61% indicating a critical SLA breach.

Team D's Friday performance (61%) and Thursday (76%) require urgent investigation. A rota adjustment or shift reinforcement for Thursdays and Fridays is recommended across all teams.

A heatmap colour-codes all 20 data points simultaneously, making row and column patterns immediately visible."""

heatmap1_spec = {
    "type": "heatmap",
    "title": "SLA Compliance (%) by Team and Day of Week",
    "x_axis": "Day of Week",
    "y_axis": "Support Team",
    "data": [
        {"row": "Team A", "col": "Mon", "value": 94},
        {"row": "Team A", "col": "Tue", "value": 96},
        {"row": "Team A", "col": "Wed", "value": 92},
        {"row": "Team A", "col": "Thu", "value": 89},
        {"row": "Team A", "col": "Fri", "value": 78},
        {"row": "Team B", "col": "Mon", "value": 88},
        {"row": "Team B", "col": "Tue", "value": 91},
        {"row": "Team B", "col": "Wed", "value": 87},
        {"row": "Team B", "col": "Thu", "value": 85},
        {"row": "Team B", "col": "Fri", "value": 72},
        {"row": "Team C", "col": "Mon", "value": 97},
        {"row": "Team C", "col": "Tue", "value": 98},
        {"row": "Team C", "col": "Wed", "value": 96},
        {"row": "Team C", "col": "Thu", "value": 94},
        {"row": "Team C", "col": "Fri", "value": 88},
        {"row": "Team D", "col": "Mon", "value": 82},
        {"row": "Team D", "col": "Tue", "value": 84},
        {"row": "Team D", "col": "Wed", "value": 80},
        {"row": "Team D", "col": "Thu", "value": 76},
        {"row": "Team D", "col": "Fri", "value": 61},
    ],
    "insights": "Universal Friday decline; Team C = consistently best; Team D Friday (61%) is critical SLA breach.",
}

# ---

radar1_query = """Vendor evaluation scorecard — Three vendors scored on 5 criteria (scale 1–10):

Criteria        Vendor X   Vendor Y   Vendor Z
Price              8          6          9
Quality            7          9          6
Support            6          8          5
Delivery Speed     9          7          8
Innovation         5          8          7

Which vendor is the best overall choice?"""

radar1_reasoning = """Three entities (vendors) scored across five criteria — a multi-dimensional comparison. A radar chart (spider chart) is ideal here because:
1. It displays multiple attributes for multiple entities on a single chart.
2. The "shape" of each entity's polygon visually encodes its strength/weakness profile.
3. A heatmap would also work (3×5 matrix) but radar charts are more intuitive for showing "balanced vs. specialised" profiles.

No single vendor dominates on all axes — this makes the trade-off visualisation especially valuable.
Confidence: HIGH — all 15 values explicit."""

radar1_answer = """No single vendor excels across all criteria. The radar chart reveals distinct profiles:
- **Vendor X**: Strong on Price (8) and Delivery (9), weak on Innovation (5) and Support (6) — best for cost-sensitive, speed-critical procurement.
- **Vendor Y**: Leads on Quality (9), Support (8), and Innovation (8), but weakest on Price (6) — best for long-term strategic partnerships where quality matters most.
- **Vendor Z**: Strongest price (9) but weakest Support (5) and Quality (6) — high risk for complex deployments.

Recommendation: Vendor Y for critical enterprise contracts; Vendor X for high-volume commodity purchases.

A radar chart makes each vendor's strength/weakness profile immediately visible and enables direct multi-criteria comparison."""

radar1_spec = {
    "type": "radar",
    "title": "Vendor Evaluation — Multi-Criteria Comparison",
    "x_axis": None,
    "y_axis": "Score (1–10)",
    "data": [
        {"entity": "Vendor X", "dimensions": {"Price": 8, "Quality": 7, "Support": 6, "Delivery Speed": 9, "Innovation": 5}},
        {"entity": "Vendor Y", "dimensions": {"Price": 6, "Quality": 9, "Support": 8, "Delivery Speed": 7, "Innovation": 8}},
        {"entity": "Vendor Z", "dimensions": {"Price": 9, "Quality": 6, "Support": 5, "Delivery Speed": 8, "Innovation": 7}},
    ],
    "insights": "Vendor X = speed/cost; Vendor Y = quality/support/innovation; Vendor Z = cheap but risky on support/quality.",
}

# ---

heatmap2_query = """Retail store — Product category sales heat by hour of day and day of week (£ 000s, one week sample):

               Mon  Tue  Wed  Thu  Fri  Sat  Sun
Morning(9-12):  8    7    9    8   11   18   15
Midday(12-15): 12   11   13   12   16   24   20
Afternoon(3-6): 9    8   10    9   14   22   18
Evening(6-9):   5    4    6    5   10   16   12

When should additional staff be rostered?"""

heatmap2_reasoning = """A 4×7 matrix of sales values (time-of-day × day-of-week). A heatmap is the most informative choice:
1. Two categorical dimensions crossed with a continuous metric is the defining use case for a heatmap.
2. With 28 cells, a table is readable but a heatmap immediately surfaces the peak and trough cells through colour.
3. Staffing decisions need to identify peak cells quickly — the heatmap makes this actionable.

Peak cell: Saturday Midday (£24k). Lowest: Tuesday Evening (£4k).
Confidence: HIGH."""

heatmap2_answer = """The heatmap reveals clear patterns:
- **Weekend premium**: Saturday and Sunday sales are 2–3× weekday levels across all time slots.
- **Midday peak**: The 12–15:00 slot is consistently the busiest period every day.
- **Saturday Midday is the single peak** at £24k — this slot should have maximum staff coverage.
- **Tuesday Evening is the trough** at £4k — minimum staffing appropriate.

Staffing recommendation: Reinforce Saturday and Sunday, particularly 12:00–18:00. Monday–Thursday evenings can operate at reduced staffing levels.

A heatmap with colour gradient instantly surfaces the high and low demand cells across the full week."""

heatmap2_spec = {
    "type": "heatmap",
    "title": "Sales by Time Slot and Day of Week (£000s)",
    "x_axis": "Day of Week",
    "y_axis": "Time of Day",
    "data": [
        {"row": "Morning (9–12)", "col": "Mon", "value": 8},
        {"row": "Morning (9–12)", "col": "Tue", "value": 7},
        {"row": "Morning (9–12)", "col": "Wed", "value": 9},
        {"row": "Morning (9–12)", "col": "Thu", "value": 8},
        {"row": "Morning (9–12)", "col": "Fri", "value": 11},
        {"row": "Morning (9–12)", "col": "Sat", "value": 18},
        {"row": "Morning (9–12)", "col": "Sun", "value": 15},
        {"row": "Midday (12–15)", "col": "Mon", "value": 12},
        {"row": "Midday (12–15)", "col": "Tue", "value": 11},
        {"row": "Midday (12–15)", "col": "Wed", "value": 13},
        {"row": "Midday (12–15)", "col": "Thu", "value": 12},
        {"row": "Midday (12–15)", "col": "Fri", "value": 16},
        {"row": "Midday (12–15)", "col": "Sat", "value": 24},
        {"row": "Midday (12–15)", "col": "Sun", "value": 20},
        {"row": "Afternoon (15–18)", "col": "Mon", "value": 9},
        {"row": "Afternoon (15–18)", "col": "Tue", "value": 8},
        {"row": "Afternoon (15–18)", "col": "Wed", "value": 10},
        {"row": "Afternoon (15–18)", "col": "Thu", "value": 9},
        {"row": "Afternoon (15–18)", "col": "Fri", "value": 14},
        {"row": "Afternoon (15–18)", "col": "Sat", "value": 22},
        {"row": "Afternoon (15–18)", "col": "Sun", "value": 18},
        {"row": "Evening (18–21)", "col": "Mon", "value": 5},
        {"row": "Evening (18–21)", "col": "Tue", "value": 4},
        {"row": "Evening (18–21)", "col": "Wed", "value": 6},
        {"row": "Evening (18–21)", "col": "Thu", "value": 5},
        {"row": "Evening (18–21)", "col": "Fri", "value": 10},
        {"row": "Evening (18–21)", "col": "Sat", "value": 16},
        {"row": "Evening (18–21)", "col": "Sun", "value": 12},
    ],
    "insights": "Peak: Sat Midday £24k; Trough: Tue Evening £4k; Weekend 2-3x weekday; midday consistently busiest slot.",
}

# ---

radar2_query = """Employee performance review — Five competency scores (1–5 scale) for three employees under review for promotion:

Competency          Alice   Bob   Carol
Technical Skills      4      5     3
Communication         5      3     4
Leadership            3      4     5
Problem Solving       4      4     4
Adaptability          5      3     5

Who should be promoted to the team lead role?"""

radar2_reasoning = """Three candidates scored on 5 competencies — multi-dimensional comparison. A radar chart is ideal for promotion decisions because:
1. The spider chart polygon shape for each candidate visually encodes their overall balance vs. specialisation.
2. For a team lead role, Leadership and Communication are likely weighted higher — the radar makes these dimensions visually prominent.
3. A table of 15 numbers is harder for a hiring panel to interpret at a glance.

All three candidates score 4 on Problem Solving — no differentiation there.
Confidence: HIGH — all 15 values explicit."""

radar2_answer = """The radar chart profiles reveal clearly differentiated candidates:
- **Alice**: Strongest in Communication (5) and Adaptability (5); solid Technical (4) and Problem Solving (4); weakest in Leadership (3). Best fit for senior individual contributor or client-facing roles.
- **Bob**: Dominant in Technical Skills (5) and Leadership (4); weakest in Communication (3) and Adaptability (3). Strong technical lead but interpersonal development needed.
- **Carol**: Strongest in Leadership (5) and Adaptability (5); Communication (4) and Problem Solving (4) are solid; Technical Skills (3) is the gap.

**Recommendation: Carol** for the Team Lead role — highest Leadership score (5), high Adaptability (5), and balanced Communication. Technical gaps can be compensated by the team. Bob should be considered for a Principal Engineer track.

A radar chart makes each candidate's full competency profile visible simultaneously."""

radar2_spec = {
    "type": "radar",
    "title": "Employee Competency Profiles — Promotion Review",
    "x_axis": None,
    "y_axis": "Score (1–5)",
    "data": [
        {"entity": "Alice", "dimensions": {"Technical Skills": 4, "Communication": 5, "Leadership": 3, "Problem Solving": 4, "Adaptability": 5}},
        {"entity": "Bob", "dimensions": {"Technical Skills": 5, "Communication": 3, "Leadership": 4, "Problem Solving": 4, "Adaptability": 3}},
        {"entity": "Carol", "dimensions": {"Technical Skills": 3, "Communication": 4, "Leadership": 5, "Problem Solving": 4, "Adaptability": 5}},
    ],
    "insights": "Carol leads on Leadership+Adaptability — best fit for Team Lead; Bob strong technically; Alice best for client-facing IC roles.",
}

# ---

heatmap3_query = """Risk register — Project risk matrix, impact vs. likelihood scores (1–5 scale):

Risk Item            Likelihood  Impact
Data Breach               3         5
Scope Creep               4         3
Key Staff Departure       2         4
Vendor Delay              4         4
Budget Overrun            3         4
Regulatory Change         2         5
Technology Failure        2         4

Create a risk visualisation and identify the priority risks."""

heatmap3_reasoning = """A risk register with two numerical dimensions (likelihood × impact) for 7 risk items. A heatmap/risk matrix is the industry-standard visualisation for this data because:
1. The 5×5 likelihood-impact grid is a universally understood risk management tool.
2. Cells in the top-right quadrant (high likelihood, high impact) are the red zone — immediately visible via colour coding.
3. Each risk item can be plotted as a data point on the grid, or the grid can be colour-coded with risks labelled in cells.

Critical risks (likelihood × impact ≥ 12): Vendor Delay (4×4=16), Regulatory Change (2×5=10, moderate).
Confidence: HIGH."""

heatmap3_answer = """Mapping all risks to the likelihood-impact matrix:
- **Critical (top-right)**: Vendor Delay (4×4=16) — highest risk score; requires immediate mitigation plan.
- **High impact, lower likelihood**: Data Breach (3×5=15) and Regulatory Change (2×5=10) — severe if realised; monitoring and contingency plans essential.
- **High likelihood, moderate impact**: Scope Creep (4×3=12) — most likely to occur; strong change control process needed.
- **Medium**: Budget Overrun (3×4=12), Key Staff Departure (2×4=8).

Priority mitigation order: Vendor Delay → Data Breach → Scope Creep → Regulatory Change.

A heatmap risk matrix, with risks plotted by their likelihood and impact coordinates, is the definitive visualisation for risk prioritisation."""

heatmap3_spec = {
    "type": "heatmap",
    "title": "Project Risk Matrix — Likelihood vs. Impact",
    "x_axis": "Impact (1–5)",
    "y_axis": "Likelihood (1–5)",
    "data": [
        {"risk": "Data Breach", "likelihood": 3, "impact": 5, "score": 15},
        {"risk": "Scope Creep", "likelihood": 4, "impact": 3, "score": 12},
        {"risk": "Key Staff Departure", "likelihood": 2, "impact": 4, "score": 8},
        {"risk": "Vendor Delay", "likelihood": 4, "impact": 4, "score": 16},
        {"risk": "Budget Overrun", "likelihood": 3, "impact": 4, "score": 12},
        {"risk": "Regulatory Change", "likelihood": 2, "impact": 5, "score": 10},
        {"risk": "Technology Failure", "likelihood": 2, "impact": 4, "score": 8},
    ],
    "insights": "Top priority: Vendor Delay (score 16); Data Breach second (15); Scope Creep highest likelihood at 4.",
}

# ─────────────────────────────────────────────
# SCENARIO 6 — When NOT to visualize (5)
# ─────────────────────────────────────────────

no_viz1_query = """Compliance report — GDPR audit findings for Q3 2024:

Finding 1: Data retention policy was last reviewed in March 2022. Current regulation requires annual review. Status: Non-compliant.
Finding 2: Consent forms updated to include new data processing purposes. Status: Compliant.
Finding 3: Data Protection Officer appointment documentation is incomplete — appointment letter not filed. Status: Non-compliant.
Finding 4: Subject Access Request log is maintained and responses issued within 30-day SLA. Status: Compliant.
Finding 5: Third-party processor agreements for 3 vendors do not include required Article 28 clauses. Status: Non-compliant.

Can you visualise this compliance status?"""

no_viz1_reasoning = """The data consists of 5 audit findings, each with a qualitative description and a binary status (Compliant/Non-compliant). Assessing whether a chart is appropriate:
1. There are only 5 data points with two possible values — a pie chart of 3 non-compliant vs 2 compliant would be technically possible but adds no insight beyond what the text already states.
2. The critical value of this data is the DESCRIPTION of each finding (what is non-compliant and why), not an aggregate count.
3. A visualisation would strip away the most important information (the finding narrative) and reduce 5 rich findings to a 3/2 split — this is a loss of fidelity, not a gain in clarity.
4. A structured table (Finding | Status | Description) is the correct presentation format.

Verdict: Do not visualise — a table preserves the essential detail; a chart would obscure it.
Confidence: HIGH."""

no_viz1_answer = """I would not recommend a visualisation for this data. Here is why:

The audit findings contain five distinct, qualitative findings with binary Compliant/Non-compliant status. Converting this to a chart — for example, a pie chart showing 3 Non-compliant vs. 2 Compliant — would discard the most critical information: the description of each finding, which is what the compliance team needs to act on.

A structured table is the correct format for this data:

| # | Finding | Status |
|---|---------|--------|
| 1 | Data retention policy overdue for annual review | Non-compliant |
| 2 | Consent forms updated for new processing purposes | Compliant |
| 3 | DPO appointment documentation incomplete | Non-compliant |
| 4 | SAR log maintained, 30-day SLA met | Compliant |
| 5 | 3 vendor agreements missing Article 28 clauses | Non-compliant |

**3 of 5 findings are non-compliant.** The two items requiring immediate action are Finding 5 (vendor agreements — affects 3 processors and carries highest regulatory risk) and Finding 3 (DPO documentation). A chart would reduce these actionable details to a meaningless 3/2 ratio and should not be used here."""

no_viz2_query = """Legal document — Key terms of a software licence agreement (extract):

1. Licence Grant: Non-exclusive, non-transferable licence for internal business use only.
2. Term: Initial 36-month term, renewable annually thereafter.
3. Permitted Users: Up to 500 named users within the licensed entity.
4. Prohibited Uses: Sublicensing, reverse engineering, resale, SaaS redistribution.
5. SLA: 99.5% uptime guarantee; 4-hour critical response; £500/hour credit for breaches exceeding 24 hours.
6. Liability Cap: Aggregate liability limited to 12 months' licence fees paid.
7. Governing Law: English law; jurisdiction of the courts of England and Wales.

Suggest a chart to summarise this licence."""

no_viz2_reasoning = """The licence agreement contains 7 heterogeneous legal terms covering different subjects: scope, duration, users, restrictions, SLA, liability, and jurisdiction. Each item is a distinct legal concept with complex qualitative content. Assessing visualisation suitability:
1. There is no numerical quantity being compared across categories — there is no "magnitude" axis.
2. There is no time series — the term is a single stated value (36 months), not a progression.
3. There is no composition (no parts summing to a whole).
4. Forcing this into a chart would require inventing an artificial numerical encoding that doesn't reflect the underlying meaning.
5. This data is inherently textual and structured — a well-formatted table or numbered summary is the correct format.

Verdict: This is not appropriate for any chart type. A structured table or bullet list is correct.
Confidence: HIGH."""

no_viz2_answer = """A chart is not appropriate for this data. The licence terms are heterogeneous legal clauses — each covers a completely different aspect of the agreement (scope, term, users, restrictions, SLA, liability, jurisdiction). There is no single quantitative dimension across which to compare them; any chart would require artificially encoding meaning that doesn't exist in the data.

The correct presentation is a structured summary table:

| Clause | Key Term |
|--------|----------|
| Licence Type | Non-exclusive, non-transferable; internal use only |
| Duration | 36 months + annual renewal |
| User Limit | 500 named users |
| Prohibited | Sublicensing, resale, SaaS redistribution, reverse engineering |
| SLA | 99.5% uptime; 4hr critical response; £500/hr credit >24hr breach |
| Liability Cap | 12 months' licence fees paid |
| Governing Law | English law; England & Wales courts |

**Key risk items** for legal review: the liability cap (12 months' fees) may be insufficient for SaaS dependency scenarios, and the 500-user limit should be verified against current and projected headcount."""

no_viz3_query = """M&A due diligence report extract — Financial highlights:

Target company: Acme Manufacturing Ltd
Revenue FY2024: £18.3 M
Revenue FY2023: £16.7 M
EBITDA FY2024: £3.2 M (17.5% margin)
EBITDA FY2023: £2.9 M (17.4% margin)
Net Debt: £4.1 M
Cash: £0.8 M
Headcount: 142 employees
Key customers: Top 3 customers account for 67% of revenue

Should I create a visualisation of this data?"""

no_viz3_reasoning = """Assessing whether a chart adds value here:
1. There are only 2 time periods (FY2023, FY2024) — a line chart with just 2 points communicates no more than the numbers themselves, and a 2-point "trend" is not meaningful.
2. The data points cover heterogeneous metrics (revenue, EBITDA, debt, cash, headcount) that cannot be meaningfully combined on a single chart.
3. Multiple small charts (one per metric) would be possible but with only 2 data points each, they add visual complexity without analytical value.
4. The customer concentration stat (67% from top 3) is a risk signal best stated as text with context rather than a pie chart.
5. For a due diligence audience, precise numbers in a structured table carry more credibility than rounded chart values.

Verdict: A structured financial summary table is more appropriate. Charts are not warranted for 2-period comparison with heterogeneous metrics.
Confidence: HIGH."""

no_viz3_answer = """I would not create a visualisation for this M&A summary data. Here is the reasoning:

1. **Too few time periods for trend analysis**: With only FY2023 and FY2024 data, a line chart would contain just two points — this communicates no more insight than the numbers themselves, and a two-point "trend" is statistically meaningless.
2. **Heterogeneous metrics**: Revenue, EBITDA, debt, cash, and headcount cannot be plotted on a single chart without artificial normalisation that could distort interpretation.
3. **Due diligence context requires precision**: M&A audiences require exact figures; rounding to chart scale reduces credibility.

The correct presentation is a concise financial summary table:

| Metric | FY2024 | FY2023 | Change |
|--------|--------|--------|--------|
| Revenue | £18.3 M | £16.7 M | +9.6% |
| EBITDA | £3.2 M | £2.9 M | +10.3% |
| EBITDA Margin | 17.5% | 17.4% | +0.1pp |
| Net Debt | £4.1 M | — | — |
| Cash | £0.8 M | — | — |

**Key risk flag**: Top 3 customer concentration at 67% of revenue is a material customer dependency risk that should be quantified and stress-tested in the due diligence model. Net debt of £4.1 M against EBITDA of £3.2 M gives a leverage ratio of ~1.3x — within normal range."""

no_viz4_query = """IT asset inventory — Server estate summary:

Server Name   | OS            | RAM  | CPU Cores | Location    | Status
SRV-PROD-001  | Ubuntu 22.04  | 64GB | 16        | DC-London   | Active
SRV-PROD-002  | Ubuntu 22.04  | 64GB | 16        | DC-London   | Active
SRV-PROD-003  | RHEL 8.6      | 128GB| 32        | DC-Manchester| Active
SRV-DB-001    | RHEL 8.6      | 256GB| 32        | DC-London   | Active
SRV-DB-002    | RHEL 8.6      | 256GB| 32        | DC-Manchester| Active
SRV-DEV-001   | Ubuntu 20.04  | 32GB | 8         | DC-London   | Decommission pending
SRV-DEV-002   | Ubuntu 20.04  | 32GB | 8         | DC-London   | Active

Can you chart this server inventory?"""

no_viz4_reasoning = """Assessing chart suitability for this server inventory:
1. The inventory is a list of 7 individual assets with multiple attributes per row — this is inherently tabular, multi-attribute data.
2. While aggregate views are possible (e.g., bar chart of RAM by server), the primary use of an inventory is to look up specific servers — charts do not support lookup queries.
3. A bar chart of CPU/RAM per server would lose the OS, location, and status attributes — destroying the most operationally critical information.
4. The data is already in perfect table format — converting it to a chart would lose fidelity without adding insight.
5. If aggregate analysis were needed (e.g., total RAM by data centre, count by OS), a chart could support that specific question — but the question asks to "chart the inventory", which is not appropriate.

Verdict: Keep as a table. Charts may be appropriate for derived aggregates if specifically asked.
Confidence: HIGH."""

no_viz4_answer = """A chart is not the right format for this server inventory data. Here is why:

An asset inventory serves as a **lookup and operational reference tool** — administrators need to find a specific server and see all its attributes at a glance. Converting this to a chart would:
1. Require plotting on a single dimension (e.g., RAM values), discarding OS version, location, and status — the operationally critical attributes.
2. Make the decommission flag for SRV-DEV-001 invisible or buried.
3. Offer no advantage for the actual use case (auditing, capacity planning, incident response).

The inventory table is already the correct format. The structured table should be preserved and enhanced with filtering/sorting capability.

**If you need derived aggregates, I can produce those with appropriate charts:**
- Total RAM capacity by data centre → bar chart (London: 448 GB, Manchester: 384 GB)
- Server count by OS → pie chart (Ubuntu 22.04: 2, Ubuntu 20.04: 2, RHEL 8.6: 3)
- Status breakdown → simple count (Active: 6, Decommission pending: 1)

Please specify which aggregate view would be useful and I will generate the chart specification for that."""

no_viz5_query = """Narrative performance review — Annual review for Senior Manager Jane Smith:

Jane consistently demonstrated strong leadership through a restructuring period that affected 45 team members. She maintained team morale by holding weekly one-to-ones and introducing a peer recognition scheme that saw 94% participation. Her communication with senior stakeholders was characterised by clarity and appropriate escalation — three critical issues were surfaced and resolved before impacting the business. On the development side, Jane completed her executive coaching programme and applied learnings visibly in two difficult client negotiations. One area for growth is delegation: Jane occasionally handled tasks that could have developed junior team members. Overall, Jane is performing above expectations and is recommended for the next level consideration.

Please visualise Jane's performance."""

no_viz5_reasoning = """Assessing whether a visualisation is appropriate for a narrative performance review:
1. The content is qualitative narrative — it contains a few quantitative anchors (45 team members, 94% recognition participation, 3 issues escalated) but they are context-setting details embedded in prose, not a dataset.
2. Forcing the narrative into a chart (e.g., a radar of competencies) would require the reviewer to subjectively score each dimension — scores that are not present in the document and would involve making up data.
3. The review's value is precisely its narrative nuance — "appropriate escalation", "difficult client negotiations", "delegation as growth area" — these cannot be encoded in chart axes without losing meaning.
4. A chart of inferred scores would misrepresent the document's content and potentially introduce bias.
5. The appropriate response is a structured narrative summary with key evidence points highlighted.

Verdict: Do not visualise. The data is narrative; charts require numerical data. Summarise as structured text.
Confidence: HIGH."""

no_viz5_answer = """A visualisation is not appropriate for this performance review. Here is the reasoning:

Performance narratives contain **qualitative assessments** — they describe behaviours, context, and developmental observations that resist reduction to numbers. Any attempt to chart this would require me to invent scores (e.g., "Leadership: 8/10") that are not present in the document. Inventing those scores would misrepresent the reviewer's intent and introduce arbitrary bias.

**What the document does tell us — structured summary:**

| Dimension | Evidence | Assessment |
|-----------|----------|------------|
| Leadership | Managed 45-person restructuring; weekly 1:1s; recognition scheme (94% uptake) | Strength |
| Stakeholder Communication | 3 critical issues surfaced and resolved before business impact | Strength |
| Development | Completed executive coaching; applied in 2 client negotiations | On track |
| Delegation | Occasionally retained tasks suitable for junior development | Growth area |
| Overall | Recommended for next-level consideration | Above expectations |

If a radar chart of competency scores is required for a formal review system, those scores should be explicitly captured by the reviewer on a defined scale — I should not infer them from narrative text. Please provide the explicit scorecard if a chart is needed."""

# ─────────────────────────────────────────────
# Assemble all examples
# ─────────────────────────────────────────────

examples = [
    # Time series — line chart
    make_example(ts1_query, ts1_reasoning, ts1_answer, ts1_spec),
    make_example(ts2_query, ts2_reasoning, ts2_answer, ts2_spec),
    make_example(ts3_query, ts3_reasoning, ts3_answer, ts3_spec),
    make_example(ts4_query, ts4_reasoning, ts4_answer, ts4_spec),
    make_example(ts5_query, ts5_reasoning, ts5_answer, ts5_spec),
    # Categorical comparison — bar chart
    make_example(bar1_query, bar1_reasoning, bar1_answer, bar1_spec),
    make_example(bar2_query, bar2_reasoning, bar2_answer, bar2_spec),
    make_example(bar3_query, bar3_reasoning, bar3_answer, bar3_spec),
    make_example(bar4_query, bar4_reasoning, bar4_answer, bar4_spec),
    make_example(bar5_query, bar5_reasoning, bar5_answer, bar5_spec),
    # Composition — pie/donut
    make_example(pie1_query, pie1_reasoning, pie1_answer, pie1_spec),
    make_example(pie2_query, pie2_reasoning, pie2_answer, pie2_spec),
    make_example(pie3_query, pie3_reasoning, pie3_answer, pie3_spec),
    make_example(pie4_query, pie4_reasoning, pie4_answer, pie4_spec),
    make_example(pie5_query, pie5_reasoning, pie5_answer, pie5_spec),
    # Distribution — histogram
    make_example(hist1_query, hist1_reasoning, hist1_answer, hist1_spec),
    make_example(hist2_query, hist2_reasoning, hist2_answer, hist2_spec),
    make_example(hist3_query, hist3_reasoning, hist3_answer, hist3_spec),
    make_example(hist4_query, hist4_reasoning, hist4_answer, hist4_spec),
    make_example(hist5_query, hist5_reasoning, hist5_answer, hist5_spec),
    # Multi-dimensional — heatmap / radar
    make_example(heatmap1_query, heatmap1_reasoning, heatmap1_answer, heatmap1_spec),
    make_example(radar1_query, radar1_reasoning, radar1_answer, radar1_spec),
    make_example(heatmap2_query, heatmap2_reasoning, heatmap2_answer, heatmap2_spec),
    make_example(radar2_query, radar2_reasoning, radar2_answer, radar2_spec),
    make_example(heatmap3_query, heatmap3_reasoning, heatmap3_answer, heatmap3_spec),
    # When NOT to visualize
    make_example(no_viz1_query, no_viz1_reasoning, no_viz1_answer, None),
    make_example(no_viz2_query, no_viz2_reasoning, no_viz2_answer, None),
    make_example(no_viz3_query, no_viz3_reasoning, no_viz3_answer, None),
    make_example(no_viz4_query, no_viz4_reasoning, no_viz4_answer, None),
    make_example(no_viz5_query, no_viz5_reasoning, no_viz5_answer, None),
]

assert len(examples) == 30, f"Expected 30 examples, got {len(examples)}"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Written {len(examples)} examples to {OUTPUT_PATH}")

# Validate: each line is valid JSON and contains required keys
with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

categories = {}
errors = []
for i, line in enumerate(lines):
    try:
        obj = json.loads(line)
        for key in ("text", "category", "difficulty", "source"):
            if key not in obj:
                errors.append(f"Line {i+1}: missing key '{key}'")
        cat = obj.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        # Check chart_spec is valid JSON if present
        text = obj["text"]
        if "<chart_spec>" in text:
            start = text.index("<chart_spec>") + len("<chart_spec>")
            end = text.index("</chart_spec>")
            spec_str = text[start:end].strip()
            json.loads(spec_str)  # will raise if invalid
    except json.JSONDecodeError as e:
        errors.append(f"Line {i+1}: JSON error — {e}")

if errors:
    print("VALIDATION ERRORS:")
    for e in errors:
        print(f"  {e}")
else:
    print("All 30 examples validated successfully.")

print(f"Category breakdown: {categories}")

# Count by chart type
chart_types = {}
for line in lines:
    obj = json.loads(line)
    text = obj["text"]
    if "<chart_spec>" in text:
        start = text.index("<chart_spec>") + len("<chart_spec>")
        end = text.index("</chart_spec>")
        spec = json.loads(text[start:end].strip())
        t = spec.get("type", "unknown")
        chart_types[t] = chart_types.get(t, 0) + 1
    else:
        chart_types["no_chart"] = chart_types.get("no_chart", 0) + 1

print(f"Chart type breakdown: {chart_types}")
