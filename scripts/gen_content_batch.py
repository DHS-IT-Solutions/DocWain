"""Generate 20 content generation training examples for DocWain (5 categories x 4 each)."""
import json, textwrap, pathlib

OUT = pathlib.Path("finetune_artifacts/sprint/claude_quality/content_gen_batch.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

SYS = ("You are DocWain, an enterprise document intelligence assistant. "
       "You analyse documents with deep contextual understanding, extract structured information, "
       "identify patterns and anomalies, and provide holistic analysis grounded in evidence. "
       "You reason step-by-step before answering, state your confidence level, and cite specific "
       "sources. When information is insufficient, you say so clearly rather than guessing.")

def ex(query, reasoning, answer):
    text = (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>")
    return {"text": text, "category": "content_generation", "difficulty": "hard", "source": "claude_quality"}

examples = []

# ── 1. Meeting notes → Email ─────────────────────────────────────────────────

examples.append(ex(
    query=textwrap.dedent("""\
    Convert these meeting notes into a professional follow-up email to all attendees.

    Meeting: Q2 Budget Review — 14 Apr 2026, 10:00–11:15
    Attendees: Sarah Chen (CFO), Raj Patel (Engineering), Lena Brooks (Marketing), Tom Wu (Sales)
    Key decisions:
    - Engineering budget increased 12% to £2.1M; headcount +2 (ML engineers)
    - Marketing campaign spend capped at £400K; digital-first strategy approved
    - Sales enablement tool renewal deferred to Q3 pending vendor negotiation
    - All departments submit revised forecasts by 22 Apr
    Action items:
    - Raj: finalise JD for ML roles by 18 Apr
    - Lena: submit digital campaign plan by 20 Apr
    - Tom: schedule vendor call week of 28 Apr
    - Sarah: circulate updated budget model by 17 Apr
    Next meeting: 28 Apr 2026, 10:00"""),
    reasoning=textwrap.dedent("""\
    The notes contain four decisions, four named action items with deadlines, and a next-meeting date.
    A follow-up email must: thank attendees, summarise decisions clearly, list each person's action
    with their deadline, and close with the next meeting. Tone should be professional but direct.
    All facts are present in the notes; no inference needed. Confidence: high."""),
    answer=textwrap.dedent("""\
    Subject: Q2 Budget Review – Decisions & Actions (14 Apr)

    Hi all,

    Thank you for a productive session this morning. Below is a summary of what was agreed.

    **Decisions**
    - Engineering budget approved at £2.1M (+12%); two ML engineer headcount additions confirmed.
    - Marketing digital-first strategy approved; campaign spend capped at £400K for Q2.
    - Sales enablement tool renewal deferred to Q3 pending vendor renegotiation.
    - All departments to submit revised forecasts by **22 April**.

    **Your Action Items**
    | Owner | Action | Due |
    |-------|--------|-----|
    | Sarah | Circulate updated budget model | 17 Apr |
    | Raj | Finalise ML engineer job descriptions | 18 Apr |
    | Lena | Submit digital campaign plan | 20 Apr |
    | Tom | Schedule vendor negotiation call | Week of 28 Apr |

    **Next Meeting:** Monday 28 April 2026, 10:00 — same dial-in link.

    Please reach out if you have questions before then.

    Best regards,
    [Your name]""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Write a follow-up email from these notes.

    Meeting: Platform Incident Post-Mortem — 13 Apr 2026
    Attendees: Dev leads, SRE team, Head of Product (Anya Reeves)
    Notes:
    - Outage lasted 47 minutes (02:14–03:01 UTC); affected EU region
    - Root cause: misconfigured autoscale policy deployed without canary
    - 1,200 enterprise tenants impacted; SLA breach for 38 tenants
    - Fix: rollback + corrected policy; monitoring alert added
    Actions:
    - SRE: publish corrected runbook by 17 Apr
    - Dev leads: mandatory canary gate in CI pipeline by 24 Apr
    - Anya: draft customer comms for 38 affected tenants by 15 Apr
    - All: review SLA credit calculations by 19 Apr"""),
    reasoning=textwrap.dedent("""\
    Incident post-mortems require factual precision—duration, scope, root cause, and fix are all
    stated explicitly. The email must not minimise impact but must be clear and forward-looking.
    SLA breach detail (38 tenants) is sensitive; include it accurately. Four action items with
    owners and dates exist in the notes. Tone: professional, accountable. Confidence: high."""),
    answer=textwrap.dedent("""\
    Subject: Platform Incident Post-Mortem – Actions & Next Steps (13 Apr)

    Hi team,

    Following today's post-mortem, here is a summary for shared reference.

    **Incident Summary**
    - **Duration:** 47 minutes (02:14–03:01 UTC, 13 Apr)
    - **Scope:** EU region; 1,200 enterprise tenants affected
    - **SLA breach:** 38 tenants — credit calculations in progress
    - **Root cause:** Autoscale policy misconfiguration deployed without canary validation
    - **Resolution:** Policy rollback applied; corrected configuration deployed; monitoring alert added

    **Action Items**
    | Owner | Action | Due |
    |-------|--------|-----|
    | Anya | Draft customer communications for 38 affected tenants | 15 Apr |
    | SRE | Publish corrected incident runbook | 17 Apr |
    | All | Complete SLA credit calculations | 19 Apr |
    | Dev leads | Enforce mandatory canary gate in CI pipeline | 24 Apr |

    The canary gate requirement is now non-negotiable for all production deployments.

    Regards,
    [Your name]""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Turn these meeting notes into a follow-up email.

    Sprint Planning — 11 Apr 2026, Team Atlas
    Attendees: PM (Dani Osei), 6 engineers, QA lead (Priya)
    Committed stories: 34 points across 9 stories
    Key decisions:
    - Auth refactor (8 pts) deprioritised to next sprint; tech-debt ticket raised
    - Data export feature (13 pts) pulled in as priority; delivery target 25 Apr
    - QA automation coverage target raised to 85% by sprint end
    - Daily standups moved to 09:30
    Action items:
    - Dani: update roadmap by 12 Apr
    - Priya: draft automation plan by 14 Apr
    - Engineers: all PRs reviewed within 24 hours policy"""),
    reasoning=textwrap.dedent("""\
    Sprint planning email needs: commitment summary, key decisions with rationale, action items,
    and logistics change (standup time). Auth refactor deferral and data export prioritisation
    are the two notable planning choices—both should be stated clearly. The PR review policy is a
    team-wide norm, not just one person's action. Confidence: high."""),
    answer=textwrap.dedent("""\
    Subject: Sprint Planning Summary – Team Atlas, Sprint starting 14 Apr

    Hi Atlas team,

    Sprint is locked. Here's what we committed to:

    **Commitment:** 34 story points / 9 stories

    **Key Decisions**
    - Auth refactor deferred to next sprint; tech-debt ticket raised for tracking.
    - Data export feature (13 pts) is the sprint priority — target delivery **25 April**.
    - QA automation coverage target: **85%** by sprint end. Priya will circulate the plan by 14 Apr.

    **Logistics**
    - Daily standups now at **09:30** starting Monday.
    - All PRs must be reviewed within **24 hours** of opening — no exceptions.

    **Actions**
    - **Dani** — update product roadmap by 12 Apr
    - **Priya** — draft QA automation plan by 14 Apr

    Let's have a strong sprint. Reach out to Dani if scope questions arise.

    Thanks,
    [Your name]""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Write a follow-up email from these board strategy meeting notes.

    Board Strategy Session — 10 Apr 2026
    Attendees: CEO (Marcus Hill), CFO, CTO, 3 Non-Exec Directors
    Notes:
    - Approved expansion into DACH market; target Q4 2026 launch
    - Rejected Series C at current valuation; revisit in 6 months
    - CTO to present AI roadmap at May board meeting
    - CFO: 18-month runway confirmed; no immediate bridge financing needed
    - Risk register updated: supply chain risk elevated to HIGH
    - Next board meeting: 12 May 2026, 09:00"""),
    reasoning=textwrap.dedent("""\
    Board meeting follow-ups are typically brief and formal. Five distinct items: market expansion
    approval, funding decision, AI roadmap request, financial status, and a risk register update.
    All decisions are explicit. Tone must be formal, concise. Distribution is to board members only
    so commercial sensitivity should be respected but content is factual. Confidence: high."""),
    answer=textwrap.dedent("""\
    Subject: Board Strategy Session — Summary & Actions (10 Apr 2026)

    Dear Board members,

    Please find below a summary of resolutions and actions from today's strategy session.

    **Resolutions**
    1. **DACH Market Expansion** — Approved. Target launch Q4 2026.
    2. **Series C Financing** — Deferred. Current valuation does not support proceeding; to be
       revisited in six months.
    3. **Financial Position** — CFO confirmed 18-month cash runway; no bridge financing required.
    4. **Risk Register** — Supply chain risk elevated to HIGH; mitigation options to be presented
       at the next session.

    **Actions**
    | Owner | Action | Due |
    |-------|--------|-----|
    | CTO | Present AI product roadmap | 12 May Board |
    | CFO | Circulate updated risk mitigation options | 12 May Board |

    **Next Board Meeting:** Tuesday 12 May 2026, 09:00.

    Papers will be distributed no later than five business days prior.

    Yours sincerely,
    [Company Secretary]""")
))

# ── 2. Data → Executive Summary ──────────────────────────────────────────────

examples.append(ex(
    query=textwrap.dedent("""\
    Write an executive summary from this Q1 2026 sales performance data.

    Region | Target (£M) | Actual (£M) | % Attainment | New Logos
    North   |    8.2      |    9.1      |    111%      |   14
    South   |    7.5      |    6.8      |     91%      |    9
    EMEA    |   12.0      |   13.4      |    112%      |   22
    APAC    |    6.0      |    4.9      |     82%      |    6
    Total   |   33.7      |   34.2      |    102%      |   51

    Top 3 deals: Nexcorp £1.8M, Titan Finance £1.2M, BlueSky £0.9M
    Average deal size: £671K (vs £520K Q1 2025, +29%)
    Churn: 3 customers, £410K ARR lost
    Pipeline entering Q2: £48.2M"""),
    reasoning=textwrap.dedent("""\
    Overall attainment is 102% — a marginal beat. EMEA and North outperformed; APAC and South
    missed. The 29% increase in average deal size is strategically significant (moving upmarket).
    Churn is modest relative to wins. Pipeline health (£48.2M) should be contextualised against
    Q2 target. An executive summary needs: headline result, regional highlights, key metrics,
    risks, and outlook. Confidence: high."""),
    answer=textwrap.dedent("""\
    **Q1 2026 Sales Performance — Executive Summary**

    **Headline:** Total revenue of £34.2M against a £33.7M target — 102% attainment, a modest
    beat driven by strong EMEA (+12%) and North (+11%) performance.

    **Regional Highlights**
    - EMEA led growth with £13.4M (112% attainment) and 22 new logos — the standout region.
    - North delivered £9.1M (111%) with 14 new logos.
    - South (91%) and APAC (82%) both missed, together accounting for £0.8M of the shortfall.

    **Deal Quality**
    Average deal size rose to £671K, up 29% year-on-year, confirming the upmarket motion is
    working. The top three deals (Nexcorp £1.8M, Titan Finance £1.2M, BlueSky £0.9M) accounted
    for £3.9M — 11% of total revenue.

    **Risk**
    Three customer churns totalling £410K ARR require root-cause review to prevent recurrence.

    **Outlook**
    Q2 pipeline stands at £48.2M — healthy coverage assuming a consistent close rate. Focus must
    shift to APAC and South recovery while protecting EMEA momentum.

    Confidence: high — all figures drawn directly from submitted data.""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Produce an executive summary from this employee engagement survey data.

    Responses: 847 of 1,100 employees (77% response rate)
    Overall engagement score: 6.8/10 (prev year: 6.4)
    Top 5 scoring areas: Team collaboration 8.2, Manager support 7.9, Role clarity 7.6,
      Tools & technology 7.4, Learning opportunities 7.1
    Bottom 5 areas: Career progression 4.9, Compensation fairness 5.1, Senior leadership
      communication 5.3, Work-life balance 5.6, Recognition 5.8
    eNPS: +18 (prev year: +7)
    Attrition risk (self-reported): 23% considering leaving within 12 months"""),
    reasoning=textwrap.dedent("""\
    Engagement improved (+0.4 pts) and eNPS nearly tripled (+18 vs +7). That's the positive
    headline. However, 23% attrition risk is a critical operational concern — nearly 1 in 4
    employees. Career progression (4.9) and compensation fairness (5.1) are the two lowest scores
    and likely drivers of attrition risk. The summary needs to balance the good news against the
    retention risk and recommend prioritised action. Confidence: high."""),
    answer=textwrap.dedent("""\
    **Employee Engagement Survey — Executive Summary (2026)**

    **Overall Result:** Engagement improved to 6.8/10 (from 6.4), with eNPS nearly tripling to
    +18. The 77% response rate provides statistically reliable data across the organisation.

    **Strengths (maintain)**
    Team collaboration (8.2) and manager support (7.9) are clear organisational strengths.
    These scores indicate a healthy immediate work environment and should be protected during
    any structural changes.

    **Critical Risks (act now)**
    - **Attrition risk: 23%** of employees self-reported likelihood of leaving within 12 months.
      At current headcount that represents ~200 people — a material operational and cost risk.
    - **Career progression (4.9)** and **compensation fairness (5.1)** are the two lowest-scoring
      areas and the most likely drivers of departure intent.
    - **Senior leadership communication (5.3)** warrants a structured internal comms review.

    **Recommended Priorities**
    1. Launch a career pathing framework with visible promotion criteria within 90 days.
    2. Commission a compensation benchmarking study against market rates.
    3. Establish a quarterly all-hands cadence with structured Q&A from senior leadership.

    Confidence: high — analysis grounded directly in survey data provided.""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Write an executive summary from this NPS and support ticket data.

    NPS score: 42 (industry benchmark: 38)
    Promoters: 58% | Passives: 26% | Detractors: 16%
    Support tickets Q1: 3,847 total
    Top 3 categories: Billing issues 1,102 (29%), Onboarding 844 (22%), API errors 611 (16%)
    Avg resolution time: 18.3 hours (SLA target: 24 hours)
    CSAT: 4.1/5.0
    Escalations: 47 (1.2% of tickets)
    Repeat contacts (same issue): 19% of tickets"""),
    reasoning=textwrap.dedent("""\
    NPS of 42 beats benchmark — positive headline. Support resolution (18.3h vs 24h SLA) is also
    healthy. However, billing issues dominating at 29% of tickets suggests a systemic product/UX
    problem, not a support problem. Repeat contact rate of 19% indicates first-contact resolution
    is poor for nearly 1 in 5 tickets. Onboarding at 22% is addressable via self-serve improvements.
    These are product and process signals, not just support metrics. Confidence: high."""),
    answer=textwrap.dedent("""\
    **Customer Experience Report — Executive Summary Q1 2026**

    **Headline:** NPS of 42 exceeds the industry benchmark of 38, and support resolution averages
    18.3 hours against a 24-hour SLA. The customer experience baseline is healthy.

    **Strengths**
    - CSAT of 4.1/5.0 reflects strong frontline support quality.
    - Escalation rate of 1.2% is low, indicating most issues are resolved at first tier.

    **Concerns Requiring Product Action**
    - **Billing (29% of tickets / 1,102):** Volume at this scale signals a product or invoicing
      UX defect — not a support staffing issue. A billing flow audit is recommended.
    - **Repeat contacts (19%):** Nearly 1 in 5 tickets requires a follow-up contact for the same
      issue, indicating root causes are not being fully resolved.
    - **Onboarding (22% of tickets):** Addressable via improved in-product guidance and
      self-serve documentation.

    **Recommendations**
    1. Product team to audit billing flow within 30 days.
    2. Support to implement structured root-cause closure on repeat-contact tickets.
    3. Customer success to review onboarding journey and expand knowledge base coverage.

    Confidence: high — all figures from submitted Q1 data.""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Produce an executive summary from this infrastructure cost data.

    Cloud spend Q1 2026: £1.24M (budget: £1.1M, +13% over budget)
    Top cost drivers:
      Compute (EC2/VMs): £612K (49%) — 34% higher than Q1 2025
      Storage: £218K (18%) — 12% higher than Q1 2025
      Data transfer: £187K (15%) — 61% higher than Q1 2025
      Managed services (DBs, queues): £143K (12%)
      Other: £80K (6%)
    Rightsizing opportunities identified: £180K annualised savings
    Reserved instance coverage: 41% (target: 70%)
    Idle resources detected: 23 instances, estimated £6K/month waste"""),
    reasoning=textwrap.dedent("""\
    Spend is 13% over budget — a meaningful variance. The dominant concern is data transfer at
    +61% YoY, which is unusual and likely driven by a specific workload change (new product feature,
    egress from a migration, etc.) — needs investigation. Compute overage is large in absolute
    terms. Two clear actionable levers exist: reserved instance coverage (41% vs 70% target) and
    idle resource cleanup. The £180K rightsizing opportunity is concrete. Confidence: high."""),
    answer=textwrap.dedent("""\
    **Cloud Infrastructure Cost Report — Executive Summary Q1 2026**

    **Headline:** Cloud spend reached £1.24M, exceeding the £1.1M budget by £140K (13%).
    Immediate optimisation levers exist that could recover approximately £180K+ annualised.

    **Key Variances**
    - **Data transfer (+61% YoY):** The most anomalous trend — a 61% year-on-year increase
      requires root-cause investigation to determine whether this is driven by growth, architectural
      changes, or unintended egress patterns.
    - **Compute (+34% YoY):** The largest absolute category (£612K). Rightsizing and reserved
      instance coverage are the primary remediation levers.

    **Optimisation Opportunities (immediate)**
    | Action | Estimated Saving |
    |--------|-----------------|
    | Rightsizing identified workloads | £180K/year |
    | Terminate 23 idle instances | £72K/year |
    | Increase RI coverage to 70% | ~£90–120K/year (estimated) |

    **Recommendations**
    1. Investigate data transfer spike — assign owner within 5 business days.
    2. Increase reserved instance coverage from 41% to 70% target before Q2 billing cycle.
    3. Terminate all 23 identified idle instances this week.

    Confidence: high — based on submitted cost breakdown data.""")
))

# ── 3. Contract → Plain Language Summary ────────────────────────────────────

examples.append(ex(
    query=textwrap.dedent("""\
    Summarise this contract clause in plain language for a non-legal audience.

    CLAUSE 14.3 — LIMITATION OF LIABILITY
    Notwithstanding any other provision of this Agreement, and to the maximum extent permitted
    by applicable law, neither Party shall be liable to the other Party for any indirect,
    incidental, special, consequential, punitive or exemplary damages, including but not limited
    to loss of profits, loss of revenue, loss of data, loss of goodwill, or cost of procurement
    of substitute goods or services, arising out of or related to this Agreement, however caused
    and regardless of the theory of liability, even if that Party has been advised of the
    possibility of such damages. Each Party's total cumulative liability arising out of or
    related to this Agreement shall not exceed the total fees paid or payable by Customer to
    Supplier in the twelve (12) month period immediately preceding the claim."""),
    reasoning=textwrap.dedent("""\
    This is a standard mutual limitation of liability clause with two components: (1) exclusion of
    indirect/consequential damages, and (2) a cap on total liability tied to 12-month fees.
    The plain-language summary must explain what each party cannot claim, what the financial cap
    means practically, and what this means for the reader — without legal jargon. I should also
    note what is NOT excluded (direct damages are implicitly allowed up to the cap).
    Confidence: high."""),
    answer=textwrap.dedent("""\
    **Plain Language Summary — Clause 14.3: Limitation of Liability**

    **What this clause says in plain English:**

    This clause limits how much either side can sue the other for if something goes wrong.

    **Two key rules:**

    1. **No claims for knock-on losses.**
       Neither party can claim for indirect or "ripple-effect" damages — for example, lost profits,
       lost customers, or the cost of finding a replacement supplier. Only direct, provable losses
       can be claimed.

    2. **There is a financial cap on claims.**
       Even for direct losses, the maximum either party can claim is capped at the total fees the
       customer paid in the 12 months before the dispute arose. If annual fees are £500K, that is
       the ceiling — regardless of actual losses.

    **What this means for you:**
    - If the supplier causes significant business disruption (e.g., a prolonged outage causing
      lost revenue), you can only recover direct costs up to the prior 12 months' fees.
    - This is a mutual cap — it applies to both sides equally.
    - This clause is standard in commercial SaaS contracts, but the cap level should be reviewed
      against your risk exposure.

    **Note:** This is a plain-language explanation only, not legal advice.""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Summarise this contract section in plain language.

    CLAUSE 9.1 — DATA PROCESSING & PRIVACY
    Customer hereby appoints Supplier as a data processor in respect of any personal data
    processed by Supplier on Customer's behalf under or in connection with this Agreement
    ("Customer Personal Data"). Supplier shall: (a) process Customer Personal Data only on
    documented instructions from Customer; (b) implement appropriate technical and organisational
    measures to protect Customer Personal Data against accidental or unlawful destruction, loss,
    alteration, unauthorised disclosure or access; (c) not engage any sub-processor without
    prior written consent of Customer; (d) notify Customer without undue delay (and in any event
    within 72 hours) upon becoming aware of a personal data breach; (e) delete or return all
    Customer Personal Data upon termination of this Agreement as directed by Customer."""),
    reasoning=textwrap.dedent("""\
    This is a GDPR-aligned data processing clause establishing the supplier as a data processor.
    Five obligations are listed (a–e). The plain-language summary must explain the customer-supplier
    relationship under GDPR, what protections are in place, the 72-hour breach notification
    requirement (which is operationally important), and what happens to data at contract end.
    The sub-processor consent right is important to flag. Confidence: high."""),
    answer=textwrap.dedent("""\
    **Plain Language Summary — Clause 9.1: Data Processing & Privacy**

    **The relationship:**
    You (the customer) own and control any personal data. The supplier only handles it on your
    behalf — they are a "data processor," which is a specific legal role under GDPR.

    **What the supplier must do:**

    1. **Follow your instructions only** — they cannot use your personal data for their own
       purposes or in ways you haven't authorised.
    2. **Keep data secure** — they must have appropriate security measures in place to protect
       against breaches, loss, or unauthorised access.
    3. **Ask permission before sharing** — they cannot pass your data to any third-party
       sub-processor (e.g., a cloud provider or analytics tool) without your written consent.
    4. **Tell you about breaches within 72 hours** — if a data breach occurs, they must notify
       you promptly. This is the same deadline GDPR imposes, so they are contractually bound.
    5. **Return or delete data when you leave** — at contract end, all your personal data must
       be returned or securely deleted as you instruct.

    **Why this matters:**
    This clause gives you meaningful control and audit rights over your data. Before signing,
    request the supplier's current sub-processor list so you know who already has consent.

    **Note:** Plain-language summary only — not legal advice.""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Translate this contract renewal clause into plain English.

    CLAUSE 22.4 — AUTOMATIC RENEWAL & TERMINATION FOR CONVENIENCE
    Unless either Party provides written notice of non-renewal to the other Party no less than
    ninety (90) days prior to the expiration of the then-current Term, this Agreement shall
    automatically renew for successive periods equal to the then-current Term (each a "Renewal
    Term"). Either party may terminate this Agreement for convenience upon one hundred and eighty
    (180) days' written notice to the other party, provided that no refund of prepaid fees shall
    be due in the event of termination for convenience by Customer."""),
    reasoning=textwrap.dedent("""\
    Two provisions here: auto-renewal (90-day opt-out window) and termination for convenience
    (180-day notice, no refund). The 90-day notice window is a common commercial trap — if missed,
    you're locked in for another full term. The no-refund clause on customer convenience termination
    is financially significant. Plain language summary must highlight the deadline risk and the
    financial consequence clearly. Confidence: high."""),
    answer=textwrap.dedent("""\
    **Plain Language Summary — Clause 22.4: Renewal & Exit Rights**

    **Auto-Renewal (the 90-day trap):**
    This contract renews automatically at the end of each term — for the same length as the
    current contract. To stop it renewing, you must send a written cancellation notice at least
    **90 days before** the contract end date. Miss that window, and you are locked in for another
    full term with no exit until the next renewal window.

    **Action required:** Set a calendar reminder at least 100 days before your contract end date.

    **Termination for Convenience:**
    Either side can exit the contract at any time by giving **180 days' written notice** — no
    reason required.

    **Important financial consequence:** If you (the customer) terminate early for convenience,
    you will **not receive a refund** of any fees you've already paid in advance. For example,
    if you've paid for 12 months and leave after month 3, you forfeit the remaining 9 months'
    fees.

    **Summary of risk:**
    - Miss the 90-day renewal window → automatic lock-in for another full term.
    - Exit mid-term → no refund on prepaid amounts.

    **Note:** Plain-language summary only — not legal advice.""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Summarise this IP ownership clause in plain language.

    CLAUSE 11 — INTELLECTUAL PROPERTY
    11.1 All intellectual property rights in the Software, Documentation and any improvements,
    modifications or derivative works thereto shall remain vested in and be the exclusive
    property of Supplier, whether or not developed using Customer's input or feedback.
    11.2 Customer shall own all intellectual property rights in Customer Data and any output,
    report or analysis generated solely from Customer Data by the Software ("Customer Output").
    11.3 Supplier is hereby granted a non-exclusive, royalty-free, worldwide, perpetual licence
    to use anonymised and aggregated Customer Data to improve Supplier's products and services,
    provided that such data does not identify Customer or any individual."""),
    reasoning=textwrap.dedent("""\
    Three sub-clauses: (1) supplier owns the software and any improvements, even if built from
    customer feedback — this is important; (2) customer owns their data and outputs from it;
    (3) supplier gets a perpetual licence to use anonymised/aggregated data for product improvement.
    Sub-clause 11.1's "even if developed using customer feedback" language is unusual and worth
    flagging. Sub-clause 11.3 is a broad licence — its scope should be understood. Confidence: high."""),
    answer=textwrap.dedent("""\
    **Plain Language Summary — Clause 11: Intellectual Property Ownership**

    **What the supplier owns (11.1):**
    The supplier owns the software, all documentation, and — critically — **any improvements or
    new features**, even if those improvements were inspired by or built on your feedback or
    suggestions. If you request a new feature and they build it, they own it.

    **What you own (11.2):**
    You own your data and any reports, outputs, or analysis that the software generates from
    your data. These are yours to use, share, or keep after the contract ends.

    **The data licence you are granting (11.3):**
    You are granting the supplier a **perpetual** (never-expiring) licence to use anonymised,
    aggregated versions of your data to improve their product. This continues after you leave.
    The protection is that the data must be anonymised and not identifiable back to you or
    any individual.

    **Watch points:**
    - If you contribute significant product input (e.g., co-development), negotiate to have
      bespoke developments carved out or jointly owned.
    - Confirm the supplier's definition of "anonymised" and their data retention period for
      this aggregated data before signing.

    **Note:** Plain-language summary only — not legal advice.""")
))

# ── 4. Technical Doc → Stakeholder Explanation ──────────────────────────────

examples.append(ex(
    query=textwrap.dedent("""\
    Explain this technical document to a non-technical executive audience.

    API Rate Limiting Implementation — Technical Specification v2.1
    Current state: No rate limiting on /api/v1/* endpoints.
    Problem: Without rate limiting, a single misconfigured client can exhaust server thread pool
    (max 500 threads), causing 503 errors for all tenants — observed in Apr 2026 incident.
    Proposed solution: Implement token bucket algorithm per tenant per endpoint.
    Parameters: 1,000 req/min burst capacity, 200 req/min sustained rate, 429 response on breach.
    Implementation: Redis-backed counter, TTL 60s, applied at API gateway layer.
    Estimated effort: 3 engineering days.
    Risk if not implemented: Repeat of Apr 2026 outage — one bad actor can take down the platform."""),
    reasoning=textwrap.dedent("""\
    Executive audience needs: what's the problem, what happened, what's the fix, what does it cost
    and what's the risk of not doing it. I should avoid "token bucket algorithm," "Redis," "TTL,"
    "thread pool" — replace with business analogies. The key message is: this is low-effort (3 days)
    with high protection value (prevents repeat outage). Confidence: high."""),
    answer=textwrap.dedent("""\
    **Rate Limiting — Stakeholder Briefing**

    **The Problem in Plain Terms**
    Our platform currently has no guardrails on how many requests any single customer can send
    per minute. In the April 2026 incident, one customer's misconfigured system flooded our
    platform with traffic and effectively blocked all other customers from accessing the service
    for the duration of the outage.

    Think of it as a single car breaking down in a tunnel and blocking all lanes — rather than
    being directed to a hard shoulder.

    **The Proposed Fix**
    We will add a "speed limit" per customer per service endpoint. Each customer gets a generous
    allowance (up to 1,000 requests per minute in bursts). If a customer exceeds their limit,
    their excess requests are politely declined with an error code — but all other customers
    are completely unaffected.

    **Cost and Timeline**
    3 engineering days to implement. No customer-facing disruption. No infrastructure cost
    beyond a minor increase in our existing caching layer.

    **Risk of Inaction**
    The April incident will recur. Any customer — even unintentionally — can repeat this
    scenario at any time. Reputational and SLA cost of another outage significantly outweighs
    3 days of engineering effort.

    **Recommendation:** Approve for immediate implementation.

    Confidence: high — based on the technical specification provided.""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Translate this technical architecture document for a non-technical board audience.

    DocWain Disaster Recovery Architecture — Summary
    Current RPO: 4 hours (Recovery Point Objective — max data loss acceptable)
    Current RTO: 8 hours (Recovery Time Objective — max downtime acceptable)
    Proposed architecture: Active-passive failover with cross-region replication to EU-West-2.
    Primary region: EU-West-1. Data replicated every 15 minutes via CDC (Change Data Capture).
    Failover trigger: Automated via health checks; manual approval required for failover execution.
    Proposed RPO: 15 minutes | Proposed RTO: 30 minutes
    Cost: £28K one-time setup, £4.2K/month ongoing
    Compliance benefit: Meets ISO 27001 Annex A.17 and NHS DSP Toolkit requirements."""),
    reasoning=textwrap.dedent("""\
    Board needs to understand the business risk reduction, not the technical implementation.
    RPO/RTO should be explained as business impact terms. The improvement is dramatic: RPO from
    4 hours to 15 minutes, RTO from 8 hours to 30 minutes. Compliance benefit is a concrete
    business value point (contractual/regulatory). Cost context: £4.2K/month vs risk of an
    8-hour outage — the ROI is clear. Confidence: high."""),
    answer=textwrap.dedent("""\
    **Disaster Recovery Upgrade — Board Briefing**

    **What Are We Protecting Against?**
    If our primary data centre fails (fire, power loss, major outage), we need a plan to restore
    service from a backup location. This proposal upgrades that plan significantly.

    **Current Vulnerability**
    Today, if our systems go down, we could lose up to **4 hours of data** and take up to
    **8 hours to restore service**. For an enterprise platform, this is a material risk.

    **What the Upgrade Delivers**
    The proposal mirrors all data to a second UK data centre (EU-West-2) every 15 minutes.
    If the primary site fails:
    - Maximum data loss: **15 minutes** (down from 4 hours)
    - Service restoration: **30 minutes** (down from 8 hours)

    **Financial Case**
    | | Cost |
    |-|------|
    | One-time setup | £28,000 |
    | Ongoing monthly | £4,200 |
    | Annual total | ~£78,000 |

    This is modest relative to the commercial cost of an 8-hour enterprise platform outage.

    **Compliance Value**
    This architecture satisfies the NHS Digital Security & Protection Toolkit and ISO 27001
    requirements — a prerequisite for several pipeline enterprise contracts.

    **Recommendation:** Approve. Low cost, high protection, compliance-enabling.

    Confidence: high — based on the architecture document provided.""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Explain this technical performance report to a non-technical product team.

    Q1 2026 Platform Performance Summary — Engineering
    P50 latency: 142ms | P95: 890ms | P99: 2,340ms
    Availability: 99.71% (SLA: 99.9%) — SLA BREACHED
    Error rate: 1.8% (target: <0.5%) — TARGET MISSED
    Slowest endpoints: /api/search (P95: 3.1s), /api/export (P95: 4.8s)
    Root cause analysis: Search uses synchronous ML inference (blocking); export lacks pagination.
    Database query time contribution: 34% of total latency
    Proposed fixes: Async inference queue for search; streaming export endpoint; DB index review.
    Estimated impact: P95 latency to ~400ms; error rate to <0.3%."""),
    reasoning=textwrap.dedent("""\
    The key messages for product team: (1) SLA was breached — this is serious and customer-facing;
    (2) two specific features (search and export) are the culprits; (3) fixes are identified and
    estimated to be impactful. Product team needs to understand customer experience impact and help
    prioritise the fixes. P50/P95/P99 need business translation. Confidence: high."""),
    answer=textwrap.dedent("""\
    **Platform Performance — Product Team Summary Q1 2026**

    **The Headline (and it's not good):**
    We breached our 99.9% availability SLA in Q1 — actual availability was 99.71%. We also missed
    our error rate target nearly 4x over. Customers on premium SLA tiers may be entitled to credits.

    **What Latency Numbers Mean for Users**
    - For the **typical user** (P50), the platform responds in 142ms — fast and unnoticeable.
    - For **1 in 20 users** (P95), they're waiting almost 1 second.
    - For **1 in 100 users** (P99), they're waiting over 2 seconds — noticeable and frustrating.

    **The Two Problem Areas**
    | Feature | Typical slow user wait | Root Cause |
    |---------|----------------------|------------|
    | Search | 3.1 seconds | AI model runs synchronously, blocking the user |
    | Export | 4.8 seconds | No streaming — waits for full dataset before responding |

    **The Fix and Expected Outcome**
    Engineering has identified solutions for both. If implemented, the engineering team estimates:
    - P95 latency drops from 890ms to ~400ms
    - Error rate drops from 1.8% to under 0.3%

    **Ask of the Product Team:**
    Prioritise the Search and Export fixes in the Q2 roadmap. These are the highest-impact
    performance improvements available.

    Confidence: high — based on Q1 engineering performance report.""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Explain this security audit finding to a non-technical CEO.

    FINDING: Critical — Unauthenticated API Endpoints
    Severity: CRITICAL (CVSS 9.1)
    Affected: /api/v1/documents/search, /api/v1/reports/generate — production environment
    Discovery: Internal penetration test, 11 Apr 2026
    Description: Two production API endpoints do not validate authentication tokens before
    processing requests. An unauthenticated attacker with network access to the API gateway can
    query and retrieve document data and generate reports on behalf of any tenant.
    Impact: Unauthorised access to all tenant document data; potential regulatory breach (GDPR).
    Remediation: Add authentication middleware to both endpoints. Estimated: 4 hours dev time.
    Status: Patch in progress; endpoints temporarily restricted to internal network only."""),
    reasoning=textwrap.dedent("""\
    This is a critical finding — unauthenticated access to tenant data is a serious GDPR and
    commercial risk. CEO needs to understand: what data is exposed, to whom, what the business
    and legal risk is, what's being done, and how quickly. The good news: discovered internally
    (not by an attacker), remediation is fast (4 hours), and interim containment is in place.
    I should be direct about the severity without causing panic, and clear about the response.
    Confidence: high."""),
    answer=textwrap.dedent("""\
    **Security Finding Briefing — CEO Summary**

    **What Was Found**
    Our internal security team discovered during routine testing that two features on our
    production platform — document search and report generation — were accessible without
    requiring a user to be logged in. In effect, someone who knew the address of these features
    could have accessed document data belonging to any customer.

    **How Serious Is This?**
    This is classified as Critical — the highest severity level. If exploited:
    - An attacker could read documents stored by any tenant on the platform.
    - This would constitute a personal data breach under GDPR, requiring notification to the
      ICO within 72 hours of confirmed exploitation.

    **Was Any Data Accessed?**
    There is currently no evidence of unauthorised access. The vulnerability was found by our
    own team, not by an external attacker.

    **Immediate Actions Taken**
    - Both features have been restricted to internal network access only while the fix is deployed.
    - The code fix is estimated at 4 hours and is in progress now.
    - We are monitoring access logs for any indication of prior exploitation.

    **What You Need to Decide**
    If log review confirms no exploitation occurred, this does not require regulatory notification.
    Legal counsel should be briefed today to confirm that position.

    Confidence: high — based on penetration test finding report provided.""")
))

# ── 5. Incident Report → Remediation Plan ───────────────────────────────────

examples.append(ex(
    query=textwrap.dedent("""\
    Write a structured remediation plan from this incident report.

    INCIDENT REPORT — INC-2026-0412
    Date: 12 Apr 2026 | Duration: 2h 14m | Severity: P1
    Affected: Payment processing service — all regions
    Impact: 4,200 transactions failed; £312K revenue not processed; 18 enterprise customers affected
    Timeline:
      14:22 — Deploy of payment-service v3.4.1
      14:31 — Error rate spike detected (12% → 67%)
      14:45 — On-call engineer paged
      15:03 — Root cause identified: DB connection pool exhaustion
      15:41 — Rollback to v3.4.0 completed
      16:36 — Full service restoration
    Root cause: v3.4.1 introduced a missing connection pool size config; default of 5 connections
    insufficient for production load. Code review did not catch the missing config.
    Contributing factors: No staging environment load test; no config validation in CI pipeline."""),
    reasoning=textwrap.dedent("""\
    The incident has a clear root cause (missing config) and two contributing factors (no load
    test, no config validation). A remediation plan needs: immediate fixes, short-term process
    improvements, and longer-term systemic changes. I should organise by timeframe and assign
    ownership types. The 45-minute detection-to-identification gap and the 38-minute rollback
    time are also worth addressing in the plan (tooling improvements). Confidence: high."""),
    answer=textwrap.dedent("""\
    **Remediation Plan — INC-2026-0412**
    *Payment Service Outage | 12 Apr 2026 | P1*

    ---

    **IMMEDIATE (Complete within 48 hours)**

    | # | Action | Owner |
    |---|--------|-------|
    | 1 | Audit all service configuration files for missing or defaulted production parameters | Platform Engineering |
    | 2 | Add connection pool size to required config checklist for payment-service | Payment Team |
    | 3 | Verify v3.4.0 configuration is correct in all regions; confirm no residual issues | SRE |
    | 4 | Begin failed transaction reprocessing for affected 4,200 transactions | Payments Ops |

    ---

    **SHORT-TERM (Complete within 2 weeks)**

    | # | Action | Owner |
    |---|--------|-------|
    | 5 | Add mandatory config schema validation step to CI/CD pipeline — blocks deploy on missing required fields | DevOps |
    | 6 | Implement load testing gate in staging for all payment-service releases | QA Lead |
    | 7 | Reduce alert-to-page time: current 23-minute gap is too long for P1 scenarios | SRE |

    ---

    **SYSTEMIC (Complete within 30 days)**

    | # | Action | Owner |
    |---|--------|-------|
    | 8 | Establish production-equivalent staging environment for payment-service | Engineering Director |
    | 9 | Implement automated rollback trigger when error rate exceeds 20% for >2 minutes | SRE |
    | 10 | Add this incident pattern to onboarding materials and code review checklist | Engineering Leads |

    ---

    **Success Criteria**
    - Zero P1 incidents caused by missing configuration in next 90 days.
    - CI pipeline blocks deploys with missing required config fields.
    - Mean time to rollback reduced to under 15 minutes.

    Confidence: high — plan derived directly from INC-2026-0412 root cause and contributing factors.""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Create a remediation plan from this data breach incident report.

    INCIDENT REPORT — SEC-2026-0408
    Date discovered: 8 Apr 2026 | Severity: Critical
    Type: Unauthorised data access via compromised service account
    Affected data: 2,300 customer records (names, emails, company names) — no financial data
    How: Former contractor's service account not deprovisioned after contract end (Jan 2026).
    Account used to query customer DB on 6 Apr 2026 at 03:14 UTC via API.
    Detection: Anomaly detection alert — off-hours API access from unrecognised IP.
    Containment: Account disabled 8 Apr 09:23; session terminated; access logs preserved.
    ICO notification: Required within 72 hours of 8 Apr (deadline: 11 Apr 2026)
    Status: ICO notified 10 Apr; customer notification in progress."""),
    reasoning=textwrap.dedent("""\
    This is a GDPR breach. Root cause is clear: failure to deprovision access for a departing
    contractor. The immediate legal obligations (ICO, customer notification) are partially
    actioned. The remediation plan must address: (1) completing notification obligations,
    (2) fixing the access management process, (3) reviewing all other dormant accounts,
    (4) technical controls to prevent recurrence. Confidence: high."""),
    answer=textwrap.dedent("""\
    **Remediation Plan — SEC-2026-0408**
    *Unauthorised Data Access — Compromised Service Account | Critical*

    ---

    **IMMEDIATE (In progress / complete by 15 Apr)**

    | # | Action | Status | Owner |
    |---|--------|--------|-------|
    | 1 | ICO notification submitted | Complete (10 Apr) | DPO |
    | 2 | Notify 2,300 affected customers per GDPR Article 34 | In progress | Legal / Comms |
    | 3 | Preserve all access logs for regulatory and forensic use | Complete | SRE |
    | 4 | Confirm no further data access occurred beyond 6 Apr event | In progress | Security |

    ---

    **SHORT-TERM (Complete within 2 weeks)**

    | # | Action | Owner |
    |---|--------|-------|
    | 5 | Audit ALL service accounts — identify and disable any without active owner | IT / Security |
    | 6 | Cross-reference contractor/employee offboarding records vs active accounts | HR + IT |
    | 7 | Implement automated account deprovisioning on contract end date | IT |

    ---

    **SYSTEMIC (Complete within 30 days)**

    | # | Action | Owner |
    |---|--------|-------|
    | 8 | Establish 30-day maximum lifetime for service accounts; renewal requires manager approval | CISO |
    | 9 | Add offboarding access revocation to HR system as mandatory checklist item | HR Director |
    | 10 | Quarterly access reviews: all privileged and service accounts certified by owner | Security |
    | 11 | Extend anomaly detection to flag all off-hours API access from new IPs in real-time | SRE |

    ---

    **Regulatory Tracking**
    - ICO case number to be recorded and monitored for response.
    - Customer complaint log to be maintained for 3 years per GDPR requirements.
    - 30-day follow-up report to ICO if required.

    Confidence: high — plan derived from SEC-2026-0408 root cause analysis.""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Write a remediation plan from this customer data loss incident report.

    INCIDENT REPORT — INC-2026-0401
    Date: 1 Apr 2026 | Duration: 6 hours | Severity: P1
    Event: Accidental deletion of production database table — document_metadata
    Cause: Engineer ran DELETE query without WHERE clause in production (intended for dev environment)
    Rows deleted: 847,000 document metadata records across 312 tenants
    Recovery: Restored from daily backup (23:00 UTC 31 Mar); 18 hours of metadata lost
    Customer impact: 312 tenants experienced missing/inaccessible documents for 6 hours
    Root cause: No query execution guard on production DB; no environment indicator in DB console;
    engineer confused production and dev credentials stored in same password manager entry."""),
    reasoning=textwrap.dedent("""\
    This is a serious data integrity incident caused by human error compounded by inadequate
    controls. Root causes: (1) no execution guard / require-WHERE enforcement on production,
    (2) no visual environment differentiation in DB tooling, (3) credential management confusion.
    The remediation plan must prevent all three. Additionally, the 18-hour data loss gap points
    to a backup frequency issue (daily is insufficient for a document platform). Confidence: high."""),
    answer=textwrap.dedent("""\
    **Remediation Plan — INC-2026-0401**
    *Accidental Production Database Deletion | P1 | 1 Apr 2026*

    ---

    **IMMEDIATE (Complete within 48 hours)**

    | # | Action | Owner |
    |---|--------|-------|
    | 1 | Validate full data restoration — confirm all 312 tenants have correct metadata | DBA |
    | 2 | Contact all 312 affected tenants with incident summary and restoration confirmation | Customer Success |
    | 3 | Enable `sql_safe_updates` mode on production DB (rejects DELETE without WHERE) | DBA |
    | 4 | Separate production and dev credentials into distinct password manager entries, clearly labelled | All engineers |

    ---

    **SHORT-TERM (Complete within 1 week)**

    | # | Action | Owner |
    |---|--------|-------|
    | 5 | Apply colour-coded environment indicators in all DB console tools (red = production) | DevOps |
    | 6 | Require second-engineer approval for any DELETE/DROP/TRUNCATE on production | Engineering Lead |
    | 7 | Increase production DB backup frequency from daily to every 2 hours | DBA |
    | 8 | Enable point-in-time recovery (PITR) on production DB to allow minute-level restoration | DBA |

    ---

    **SYSTEMIC (Complete within 30 days)**

    | # | Action | Owner |
    |---|--------|-------|
    | 9 | Implement read-only production DB access by default; write access requires explicit elevation | Security / DBA |
    | 10 | Add destructive query simulation to all-hands training; update runbooks | Engineering Director |
    | 11 | Implement DB activity monitoring with alerting on bulk delete operations (>1,000 rows) | SRE |

    ---

    **Success Criteria**
    - Production DB enforces WHERE clause requirement on all DELETE statements.
    - Backup RPO reduced to 2 hours maximum.
    - Zero incidents of production/dev environment confusion in next 12 months.

    Confidence: high — plan addresses all three root causes identified in INC-2026-0401.""")
))

examples.append(ex(
    query=textwrap.dedent("""\
    Produce a remediation plan from this supply chain incident report.

    INCIDENT REPORT — OPS-2026-0405
    Date: 5 Apr 2026 | Severity: P2
    Event: Third-party dependency (log4py v2.1.3) introduced malicious code via compromised
    maintainer account — supply chain attack. Discovered via GitHub security advisory.
    Affected systems: Document processing worker (uses log4py for audit logging)
    Exposure window: 28 Mar – 5 Apr 2026 (8 days)
    Code analysis: Malicious code attempted to beacon to external IP; firewall blocked outbound.
    Data exfiltration: None confirmed — egress blocked. No customer data accessed.
    Containment: Dependency removed; replaced with internal logging library same day.
    Affected library version pinned and blocked in package registry."""),
    reasoning=textwrap.dedent("""\
    Supply chain attacks require remediation at three levels: (1) immediate containment (done),
    (2) forensic confirmation of no exfiltration, (3) process hardening to prevent future
    supply chain attacks. The 8-day exposure window before detection is the key gap — dependency
    monitoring was insufficient. The firewall blocking egress was the saving grace. The plan
    should focus on improving dependency vetting, monitoring, and reducing future exposure windows.
    Confidence: high."""),
    answer=textwrap.dedent("""\
    **Remediation Plan — OPS-2026-0405**
    *Supply Chain Attack via Compromised Third-Party Dependency | P2 | 5 Apr 2026*

    ---

    **IMMEDIATE (Complete / in progress)**

    | # | Action | Status | Owner |
    |---|--------|--------|-------|
    | 1 | Malicious dependency removed; replaced with internal library | Complete | Platform Eng |
    | 2 | Affected version blocked in internal package registry | Complete | DevOps |
    | 3 | Full forensic review of all outbound network calls during 28 Mar–5 Apr | In progress | Security |
    | 4 | Confirm no data exfiltration (supplement firewall log review with endpoint analysis) | In progress | Security |

    ---

    **SHORT-TERM (Complete within 2 weeks)**

    | # | Action | Owner |
    |---|--------|-------|
    | 5 | Audit all third-party dependencies across all services — flag unmaintained or high-risk | Platform Eng |
    | 6 | Subscribe all repositories to GitHub/OSV security advisories with automated alerting | DevOps |
    | 7 | Implement software composition analysis (SCA) in CI pipeline — blocks on critical advisories | DevOps |

    ---

    **SYSTEMIC (Complete within 30 days)**

    | # | Action | Owner |
    |---|--------|-------|
    | 8 | Define approved dependency policy: max advisory age, maintainer activity, licence requirements | CISO |
    | 9 | Reduce dependency surface: audit and remove unused libraries across all services | All teams |
    | 10 | Implement egress allowlist at firewall level — default-deny for all new services | SRE / Security |
    | 11 | Add supply chain attack scenario to annual security training | CISO |

    ---

    **Key Finding**
    The egress firewall was the sole control that prevented data exfiltration. This must not
    be relied on as the primary defence. The 8-day detection window must be reduced to hours.

    Confidence: high — plan addresses root causes identified in OPS-2026-0405.""")
))

# ── Write JSONL ──────────────────────────────────────────────────────────────
with open(OUT, "w", encoding="utf-8") as f:
    for e in examples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"Written {len(examples)} examples to {OUT}")

# Verify
with open(OUT) as f:
    lines = f.readlines()
print(f"Lines in file: {len(lines)}")
max_len = max(len(l) for l in lines)
print(f"Max line length: {max_len} chars")
cats = {}
for l in lines:
    d = json.loads(l)
    cats[d.get("category","?")] = cats.get(d.get("category","?"), 0) + 1
print(f"Categories: {cats}")
