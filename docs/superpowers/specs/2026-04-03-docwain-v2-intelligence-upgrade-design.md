# DocWain V2 Intelligence Upgrade — Design Specification

## Overview

Upgrade DHS/DocWain from a document retrieval assistant to a GPT-class document intelligence model. Claude Code acts as researcher, teacher, and evaluator — iteratively generating training data, training the model, evaluating outputs, analyzing weaknesses, and retraining until quality gates pass across all capability tracks.

**Base model:** DHS/DocWain (Qwen3-14B, LoRA fine-tuned)
**Hardware:** NVIDIA A100-SXM4-80GB
**Training framework:** Unsloth LoRA
**Versioning:** V1 frozen, V2 promoted to `latest` only after all gates pass

## Goals

Make DHS/DocWain the "one true document analysis model" that can:

- **See** — understand layout, tables, forms, handwriting, degraded scans
- **Read** — extract text with structure preservation and semantic understanding
- **Reason** — multi-hop inference, contradiction resolution, causal analysis, quantitative reasoning
- **Connect** — leverage knowledge graph for entity disambiguation, cross-doc linking, relationship traversal
- **Visualize** — judge when data needs a chart, select the right type, produce accurate JSON specs
- **Communicate** — calibrated confidence, clear uncertainty, grounded citations, natural conversation

## Critical Requirements

### 1. Excel & CSV File Support

DocWain currently has no support for Excel (.xlsx, .xls) or CSV files. The V2 pipeline must:
- Parse Excel files preserving sheet structure, named ranges, formulas (as computed values), merged cells, and multi-sheet relationships
- Parse CSV files with intelligent delimiter detection (comma, tab, semicolon, pipe)
- Preserve tabular semantics — header detection, data type inference (dates, currency, percentages, numbers), and column relationships
- Handle large spreadsheets (100K+ rows) via chunking with context preservation
- Train the model to reason about spreadsheet data the same way it reasons about document tables

### 2. Document Processing Pipeline — Structure, Context & Insights Capture

During document processing, the pipeline must accurately capture:
- **Structure:** headings hierarchy, sections, tables, lists, figures, captions, footnotes, cross-references, page layout
- **Context:** document type classification, domain detection, purpose inference, temporal context (dates, versions, amendments)
- **Insights:** key entities, relationships, numerical patterns, anomalies, gaps, and actionable observations
- All three are extracted during ingestion and stored as enriched metadata — not raw content — for model consumption

### 3. Privacy-First Design — Metadata & Patterns Only

DocWain does not store or capture user data. For model understanding:
- Training data is exclusively synthetic — zero real document content
- The model learns from document metadata patterns: field schemas, layout patterns, entity distributions, structural conventions
- User feedback signals (corrections, low-confidence flags) are captured as anonymized patterns, never raw content
- All training examples use fabricated but realistic document structures

### 4. Comprehensive Document Understanding — Zero Information Loss

The model must efficiently:
- Identify document type (contract, invoice, report, form, policy, letter, resume, etc.) and adapt extraction strategy accordingly
- Detect and extract ALL key information — no silent omissions
- Flag sections it cannot confidently parse rather than skipping them
- Handle multi-section documents where context from section A is required to understand section B
- Perform completeness checks: compare extracted entities/fields against expected schema for the detected document type

### 5. High-Quality OCR — Images, Diagrams & Handwriting

The model must provide enterprise-grade OCR:
- **Printed text:** High-accuracy extraction from scanned documents, including degraded/skewed/low-resolution scans
- **Images & diagrams:** Identify embedded images and diagrams, extract captions, labels, axis titles, legend text, and describe visual content semantically
- **Handwritten text:** Recognize handwritten annotations, signatures, margin notes, form field entries, and mixed print+handwriting
- **Tables in images:** Reconstruct table structure from scanned/photographed tables with cell boundary detection
- **Stamps & watermarks:** Identify and extract text from stamps, watermarks, and overlaid elements without corrupting underlying text

## Non-Goals

- Image/pixel generation (model outputs structured specs, not images)
- Real-time streaming chart updates
- Training on raw customer document content (synthetic data only)
- Replacing the retrieval pipeline (model enhances it, doesn't replace it)

---

## Architecture

### Response Output Format

The model produces a structured output with three sections:

```
<think>
[Internal reasoning chain]
- What is the user asking?
- What data do I have?
- Does this need a chart? Why/why not?
- What KG context is relevant?
</think>

<response>
[Markdown-formatted answer with evidence grounding and citations]
</response>

<chart_spec>
[JSON chart specification — ONLY when visualization adds value]
</chart_spec>
```

The `<chart_spec>` block is omitted entirely when no visualization is needed. The model learns this is the common case (~70-80% of responses).

### Chart Spec JSON Schema

```json
{
  "charts": [
    {
      "id": "chart_1",
      "type": "bar|line|pie|scatter|heatmap",
      "title": "Revenue by Quarter",
      "subtitle": "FY2025 vs FY2024",
      "x": {
        "label": "Quarter",
        "values": ["Q1", "Q2", "Q3", "Q4"]
      },
      "series": [
        {
          "name": "FY2025",
          "values": [120, 145, 132, 178],
          "color": null
        },
        {
          "name": "FY2024",
          "values": [98, 112, 125, 140],
          "color": null
        }
      ],
      "unit": "$K",
      "annotations": [
        {"point": "Q3", "text": "23% YoY increase"}
      ],
      "source": "Annual Report p.14, Table 3"
    }
  ],
  "layout": "single|side_by_side|stacked"
}
```

Design decisions:
- `charts` is an array — supports up to 2-3 charts per response for comparative analysis
- `series` is an array — handles multi-series natively (overlaid datasets, grouped bars)
- `source` field — every chart traces to document evidence
- `annotations` — model highlights key data points, outliers, trends
- `color` is nullable — UI assigns theme defaults, model overrides only when semantically meaningful
- No Plotly/HTML in model output — pure JSON spec, UI renders natively

### KG Context Input Format

The model learns to consume KG context injected by the retrieval pipeline:

```
<kg_context>
entities:
  - id: E1, name: "Acme Corp", type: Organization, doc_sources: [doc_1, doc_3]
  - id: E2, name: "John Smith", type: Person, role: "CFO", doc_sources: [doc_1]
relationships:
  - E2 --[WORKS_AT]--> E1, since: 2020, source: doc_1
  - E2 --[SIGNED]--> Contract_445, date: 2024-03-15, source: doc_3
</kg_context>
```

### UI Integration

**Stack:** React 18 + TypeScript, Recharts + Nivo (already installed in docwain-ui)

**API response structure — extended:**

```typescript
{
  answer: {
    response: string,           // markdown text
    chart_spec?: {              // present only when visualization needed
      charts: Array<{
        id: string,
        type: "bar" | "line" | "pie" | "scatter" | "heatmap",
        title: string,
        subtitle?: string,
        x: { label: string, values: string[] },
        series: Array<{ name: string, values: number[], color?: string }>,
        unit?: string,
        annotations?: Array<{ point: string, text: string }>,
        source: string
      }>,
      layout: "single" | "side_by_side" | "stacked"
    },
    sources?: Array<{
      source_id: number,
      source_name: string,
      excerpt: string,
      relevance_score: number
    }>
  }
}
```

**Rendering layout inside AI message bubble:**

1. Markdown response (existing MarkdownRenderer)
2. Separator
3. Chart card (white background for readability against #416a93 bubble)
   - Recharts/Nivo render from chart_spec JSON
   - Source attribution below chart
4. Sources section
5. Feedback buttons

No raw HTML injection, no iframes. Native React component rendering from JSON spec.

---

## Training Strategy — Claude-as-Teacher Iterative Loop

### Core Loop

```
Generate seed data (2K) → Train LoRA → Evaluate (Claude Judge)
        ↑                                       │
        │              Analyze failures          │
        │              Categorize gaps           │
        │              Target weaknesses         │
        └───────────────────────────────────────┘
```

Each iteration:
1. Claude Code generates synthetic training data (SFT + DPO pairs)
2. Train with Unsloth LoRA on A100 80GB
3. Claude Code evaluates model output using scoring rubrics (1-5 scale)
4. Analyze failure patterns — which categories underperform?
5. Generate targeted data for weak spots (not just more of the same)
6. DPO pairs from actual model mistakes — rejected = what model produced, chosen = correct
7. Retrain with augmented dataset
8. Repeat until quality gates pass

### No Iteration Cap — Strategy Evolution

There is no maximum iteration limit. If gates don't pass after 5 rounds, Claude Code evolves the strategy:

- Analyzes ALL previous rounds for persistent error patterns
- Evaluates whether training data format is the problem
- Considers hyperparameter adjustments (LR, LoRA rank, batch size)
- Restructures curriculum if needed
- Shifts data generation approach (more diverse, different corruption types, deeper reasoning chains)
- Continues with the new strategy until gates pass

### Research-Informed Data Generation

Before generating data for each track, Claude Code researches:

| Track | Research Focus |
|-------|---------------|
| Excel/CSV | Spreadsheet QA benchmarks, tabular reasoning datasets, multi-sheet relationship patterns |
| Layout | DocVQA, PubLayNet, TableBank — structures that trip up document models |
| OCR & Vision | TrOCR, IAM Handwriting, diagram understanding, stamp/watermark detection literature |
| Context & Reasoning | Chain-of-thought literature, multi-hop QA patterns, reasoning distillation |
| KG-Augmented | KGQA benchmarks, GraphRAG patterns, structured knowledge integration |
| Visualization | ChartQA, PlotQA — chart generation reliability, common failure modes |

Sources: HuggingFace model cards, arXiv papers, benchmark datasets, top-performing model training recipes.

---

## Training Tracks

### Track 1: Excel & CSV Intelligence (NEW)

**Current state:** No support — DocWain cannot process spreadsheet files
**Gap:** Enterprise documents frequently include Excel/CSV data that must be analyzed

**Seed data (2.5K examples):**

| Category | Count | Purpose |
|----------|-------|---------|
| Single-sheet tabular QA | 400 | Query data from a single worksheet — lookups, filters, aggregations |
| Multi-sheet reasoning | 350 | Cross-reference data across sheets ("Sheet1.Revenue vs Sheet2.Costs") |
| Formula-aware understanding | 300 | Understand computed values, SUM/AVERAGE/VLOOKUP semantics |
| Merged cell & named range handling | 250 | Header spans, named ranges as semantic labels |
| CSV delimiter & encoding detection | 200 | Tab/semicolon/pipe delimited, UTF-8/Latin-1, quoted fields |
| Large spreadsheet chunking | 200 | 100K+ rows — answer queries without full context, chunk-aware |
| Data type inference | 250 | Dates, currencies, percentages, phone numbers — semantic typing |
| Spreadsheet-to-insight | 300 | "Summarize this spreadsheet" → structure + key patterns + anomalies |
| Negatives & edge cases | 250 | Empty sheets, pivot tables, charts-as-images, password-protected |

**The model learns to handle spreadsheet input format:**
```
<spreadsheet source="Q3_Financial_Report.xlsx">
  <sheet name="Revenue" rows="450" cols="12">
    <headers>Month | Product | Region | Units | Price | Revenue | YoY_Change</headers>
    <sample_rows>
      Jan | Widget-A | North | 1200 | $45.00 | $54,000 | +12%
      Jan | Widget-B | South | 890 | $72.50 | $64,525 | -3%
      ...
    </sample_rows>
    <summary>450 rows, 12 columns. Revenue range: $12K-$98K. 3 products, 4 regions.</summary>
  </sheet>
  <sheet name="Costs" rows="200" cols="8">
    ...
  </sheet>
</spreadsheet>
```

**Eval dimensions:** tabular_qa_accuracy, cross_sheet_reasoning, data_type_correctness, aggregation_accuracy
**Gate:** ≥ 4.0 average across all dimensions

### Track 2: Layout Intelligence (enhancing Phase 2)

**Current state:** 20K examples covering tables, layout, OCR, cross-doc
**Gap:** Simple synthetic layouts — real documents are messy. Model sometimes silently omits sections.

**Seed data (2.5K examples):**

| Category | Count | Purpose |
|----------|-------|---------|
| Nested tables (3+ levels) | 300 | Tables inside tables inside forms |
| Merged cell reasoning | 300 | Spanning headers, row groups, implicit relationships |
| Multi-column with interruptions | 250 | Figures, callout boxes, footnotes breaking flow |
| Mixed form + prose | 250 | Government forms with interleaved instructions |
| Page-spanning structures | 200 | Tables breaking across pages, continued headers |
| Hierarchical headings | 200 | 5+ level heading trees, numbering schemes |
| Document type adaptation | 300 | Contract vs invoice vs report → different extraction strategy |
| Completeness verification | 400 | Extract ALL fields, flag missed sections, compare against doc type schema |
| Edge cases & negatives | 300 | Ambiguous layouts — model should express uncertainty, never skip silently |

**Key training principle — zero information loss:**
- Every example includes a completeness check: "Expected fields for [doc_type]: [...]. Extracted: [...]. Missing: [...]"
- Model learns to flag sections it cannot parse confidently rather than silently omitting them
- DPO pairs specifically target silent omission as the rejected behavior

**Eval dimensions:** structure_accuracy, relationship_extraction, noise_robustness, completeness_score
**Gate:** ≥ 4.0 average across all dimensions

### Track 3: OCR & Vision Intelligence (enhancing Phase 2 vision)

**Current state:** Basic OCR via SigLIP + Phase 1 alignment
**Gap:** Poor handwriting recognition, no diagram understanding, no stamp/watermark handling

**Seed data (2.5K examples):**

| Category | Count | Purpose |
|----------|-------|---------|
| Printed text — clean scans | 200 | Baseline high-accuracy extraction |
| Printed text — degraded scans | 350 | Skewed, low-res, bleed-through, faded ink |
| Handwritten text — block letters | 300 | Form field entries, printed-style handwriting |
| Handwritten text — cursive/notes | 250 | Margin annotations, meeting notes, signatures |
| Mixed print + handwriting | 200 | Forms with printed labels and handwritten values |
| Diagram understanding | 300 | Flowcharts, org charts, process diagrams → semantic description |
| Chart-in-image extraction | 200 | Bar/line/pie charts in scans → extract data points and labels |
| Table-in-image reconstruction | 250 | Photographed/scanned tables → structured row/column output |
| Stamps, watermarks, overlays | 200 | Extract overlay text without corrupting underlying content |
| Caption & label extraction | 250 | Figure captions, axis titles, legend entries, callout text |

**Key training principle — see and describe:**
- For diagrams, model outputs both extracted text AND semantic description ("This is an org chart showing 3 levels of hierarchy with CEO at top...")
- For charts-in-images, model extracts approximate data values, not just labels
- Confidence scores per OCR region — model indicates which parts it's uncertain about

**Eval dimensions:** printed_accuracy, handwriting_accuracy, diagram_understanding, image_table_reconstruction, overlay_handling
**Gate:** ≥ 4.0 average across all dimensions

### Track 4: Context & Reasoning (enhancing Phase 3.5/3.7)

**Current state:** 14K examples across insights + holistic reasoning
**Gap:** Shallow synthesis — model summarizes rather than reasons

**Seed data (2K examples):**

| Category | Count | Purpose |
|----------|-------|---------|
| Multi-document contradiction resolution | 300 | Two docs disagree — which to trust and why |
| Temporal reasoning | 300 | Version-aware answers ("as of the latest amendment") |
| Implicit intent decomposition | 300 | Vague query → structured analytical plan |
| Causal chain reasoning | 250 | Multi-hop "why did X happen?" |
| Quantitative reasoning | 250 | Computing percentages, deltas, aggregations |
| Counterfactual analysis | 150 | "What if clause 4.2 were removed?" |
| Uncertainty calibration | 250 | Expressing partial knowledge with grounded confidence |
| Refusal with explanation | 200 | "I cannot determine this because..." |

**Eval dimensions:** reasoning_depth, evidence_grounding, synthesis_coherence
**Gate:** ≥ 4.0 average across all dimensions

### Track 5: KG-Augmented Knowledge (NEW)

**Current state:** Not trained — model doesn't leverage KG context
**Gap:** Model receives KG entities/relationships but ignores them

**Seed data (2K examples):**

| Category | Count | Purpose |
|----------|-------|---------|
| Entity-aware answering | 400 | KG entities to disambiguate (which "John Smith"?) |
| Relationship traversal | 350 | "Who reports to the CFO?" using org-graph edges |
| Cross-doc entity linking | 300 | Same entity across documents, consolidated answer |
| KG-grounded fact checking | 250 | Verify claims against KG triples |
| Missing relationship detection | 200 | "No relationship found between X and Y" |
| Ontology-aware reasoning | 250 | Domain-specific relationship types |
| KG context format training | 250 | Learning `<kg_context>` input format and citation |

**Eval dimensions:** entity_usage, relationship_reasoning, citation_accuracy
**Gate:** ≥ 3.8 average across all dimensions

### Track 6: Visualization Intelligence (NEW — Phase 3.8)

**Current state:** Basic `<viz>` directives in Phase 3 training
**Gap:** No structured chart_spec generation, no auto-detect judgment

**Seed data (2K examples):**

| Category | Count | Purpose |
|----------|-------|---------|
| Single-series bar/line/pie | 400 | Basic chart spec from tabular data |
| Multi-series comparison | 350 | Overlaid datasets, grouped bars, before/after |
| Auto-detect triggers | 300 | Response has chartable data → model adds chart_spec |
| Explicit request handling | 250 | User says "chart this" → generate spec |
| No-chart negatives | 400 | Data present but chart adds no value → no chart_spec |
| Annotation intelligence | 150 | Highlighting outliers, key points, trends |
| Chart type selection reasoning | 150 | Why bar vs line vs pie for this data |

**Chart trigger rules the model learns:**
- Explicit request: always generate chart_spec
- Auto-detect: generate when response contains 3+ comparable numeric values, trend data, percentage breakdowns, or ranked data
- Suppress: greetings, short answers (<100 chars), single values, non-numeric responses, gap/error responses

**Eval dimensions:** trigger_judgment, spec_correctness, data_accuracy, type_selection
**Gate:** ≥ 4.0 average across all dimensions

---

## Model Versioning

```
Ollama Registry:
  DHS/DocWain:v1       ← Current production model (FROZEN, never modified)
  DHS/DocWain:v2-wip   ← Work-in-progress during training iterations
  DHS/DocWain:latest   ← Promoted only after ALL 6 track gates pass
```

**Lifecycle:**

1. Tag current model as `DHS/DocWain:v1` before any training begins
2. Each successful training round saves checkpoint as `DHS/DocWain:v2-wip`
3. After all 6 track gates pass → run full regression suite against V1
4. Promotion gate: V2 must score ≥ V1 on ALL existing capabilities + pass all new track gates
5. Promote `v2-wip` → `latest`
6. V1 preserved indefinitely — rollback is one tag switch

**No capability regression allowed.** V2 must be a strict superset of V1's abilities.

---

## Training Execution Order

Sequential — each track builds on the previous LoRA:

```
 1. Preserve V1              → Tag DHS/DocWain:v1
 2. Track 1: Excel/CSV       → Iterative loop until gate passes
 3. Track 2: Layout          → Iterative loop until gate passes
 4. Track 3: OCR & Vision    → Iterative loop until gate passes
 5. Track 4: Context         → Iterative loop until gate passes
 6. Track 5: KG-Augmented    → Iterative loop until gate passes
 7. Track 6: Visualization   → Iterative loop until gate passes
 8. Cross-track eval         → All 6 capabilities tested together
 9. Regression vs V1         → No capability loss confirmed
10. Merge & Quantize         → GGUF Q4_K_M
11. Promote                  → DHS/DocWain:latest
```

Tracks are sequential because each builds on the previous: Excel/CSV gives tabular reasoning, layout builds structural understanding, OCR adds vision quality, reasoning builds on all extraction capabilities, KG connects entities across documents, and visualization builds on everything to present insights.

---

## Evaluation Infrastructure

### Claude Code as Judge — Scoring Rubrics

Each evaluation uses Claude Code to score model outputs on a 1-5 scale per dimension:

**1 (Poor):** Completely wrong or missing
**2 (Below):** Partially correct but major errors
**3 (Adequate):** Correct but shallow or missing nuance
**4 (Good):** Correct, well-reasoned, minor improvements possible
**5 (Excellent):** Expert-level output, fully grounded, insightful

### Track-Specific Gates

| Track | Dimensions | Pass Threshold |
|-------|-----------|----------------|
| Excel/CSV | tabular_qa_accuracy, cross_sheet_reasoning, data_type_correctness, aggregation_accuracy | ≥ 4.0 avg |
| Layout | structure_accuracy, relationship_extraction, noise_robustness, completeness_score | ≥ 4.0 avg |
| OCR & Vision | printed_accuracy, handwriting_accuracy, diagram_understanding, image_table_reconstruction, overlay_handling | ≥ 4.0 avg |
| Context | reasoning_depth, evidence_grounding, synthesis_coherence | ≥ 4.0 avg |
| KG | entity_usage, relationship_reasoning, citation_accuracy | ≥ 3.8 avg |
| Visualization | trigger_judgment, spec_correctness, data_accuracy, type_selection | ≥ 4.0 avg |

### Regression Suite

500 frozen examples covering all V1 capabilities. V2 must pass ≥ 90% of these with no category dropping below 85%.

### Cross-Track Integration Eval

150 examples that require multiple capabilities simultaneously:
- Excel data extraction + reasoning about computed values + chart generation
- Layout extraction + reasoning about extracted data
- Degraded scan OCR + entity extraction + KG linking
- Multi-doc reasoning + chart comparison
- KG context + visualization of entity relationships
- Handwritten form → OCR → layout → completeness check → insight generation
- Spreadsheet + document cross-reference (e.g., invoice data in Excel vs contract terms in PDF)

---

## File Structure

### New Files

```
src/finetune/v2/
├── data_generator/
│   ├── track1_excel_csv.py        # Excel/CSV intelligence data generator
│   ├── track2_layout.py           # Layout intelligence data generator
│   ├── track3_ocr_vision.py       # OCR & vision intelligence data generator
│   ├── track4_reasoning.py        # Context & reasoning data generator
│   ├── track5_kg.py               # KG-augmented knowledge data generator
│   ├── track6_visualization.py    # Visualization intelligence data generator
│   └── master_generator.py        # Orchestrates all tracks
├── train_track1_excel_csv.py      # Excel/CSV training script
├── train_track2_layout.py         # Layout training script
├── train_track3_ocr_vision.py     # OCR & vision training script
├── train_track4_reasoning.py      # Reasoning training script
├── train_track5_kg.py             # KG training script
├── train_track6_visualization.py  # Visualization training script
├── eval/
│   ├── rubrics_v2.py              # Judge rubrics for all 6 tracks
│   ├── cross_track_eval.py        # Integration evaluation
│   └── regression_v1.py           # V1 regression suite
├── iterative_loop.py              # Main iterative training orchestrator
└── strategy_evolver.py            # Analyzes failures, evolves training strategy
```

### New Files — Document Processing

```
src/extraction/
├── excel_parser.py                # Excel/CSV parsing with sheet structure, formulas, merged cells
├── csv_parser.py                  # CSV with intelligent delimiter/encoding detection
└── spreadsheet_chunker.py         # Large spreadsheet chunking with context preservation
```

### Modified Files

```
src/generation/prompts.py          # Updated system prompt with chart_spec format + spreadsheet context
src/ask/pipeline.py                # Parse chart_spec from model output, include in API response
src/visualization/chart_renderer.py # Template engine: chart_spec JSON → Plotly HTML (backend fallback)
src/extraction/engine.py           # Add Excel/CSV engines to extraction pipeline
src/extraction/triage.py           # Document type detection for spreadsheets
```

### UI Changes (docwain-ui repo, develop branch)

```
src/components/Chart/
├── ChartRenderer.tsx              # Renders chart_spec JSON via Recharts/Nivo
├── ChartCard.tsx                  # White card container with source attribution
└── chartUtils.ts                  # Type definitions, color themes, spec validation
src/pages/Customer/SelfAssistance/
└── ChatWindow/ChatWindow.tsx      # Extended to render chart_spec in message bubble
```

---

## Success Criteria

| Metric | V1 Baseline | V2 Target |
|--------|------------|-----------|
| Document extraction F1 | ~0.70 | ≥ 0.90 |
| Table extraction F1 | ~0.65 | ≥ 0.85 |
| Excel/CSV QA accuracy | N/A | ≥ 4.0 (judge) |
| Printed OCR accuracy | ~0.85 | ≥ 0.95 |
| Handwriting OCR accuracy | N/A | ≥ 0.75 |
| Diagram understanding | N/A | ≥ 4.0 (judge) |
| Completeness score (zero info loss) | ~0.70 | ≥ 0.90 |
| Hallucination rate | ~0.15 | ≤ 0.05 |
| Reasoning depth (judge score) | ~2.5 | ≥ 4.0 |
| KG entity usage accuracy | N/A | ≥ 3.8 (judge) |
| Chart trigger precision | N/A | ≥ 0.85 |
| Chart spec correctness | N/A | ≥ 0.90 |
| V1 regression pass rate | N/A | ≥ 90% |
| Inference speed | ~15 tok/s | ≥ 20 tok/s |

---

## Privacy & Data Policy

- All training data is synthetic — zero raw document content in training
- Model learns from metadata patterns (field schemas, layout structures, entity distributions) — never from user data
- User feedback signals captured as anonymized patterns only (e.g., "low confidence on table extraction" not the actual table)
- No PII, sensitive fields, or document content enters the training pipeline
- Extraction outputs stored as enriched metadata in Azure Blob — MongoDB is control plane only

---

## Constraints

- All training data is synthetic — zero raw document content in training
- No fixed iteration cap — strategy evolves until gates pass
- V1 is never modified — preserved as rollback safety net
- Sequential track training — no parallel LoRA merging
- A100 80GB is the target hardware for both training and inference
- No user data stored or used for training — metadata and patterns only
