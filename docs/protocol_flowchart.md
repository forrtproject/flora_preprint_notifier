# FLoRA-Notify: Preprint Flow Through the Trial

## Part 1 — Eligibility Pipeline

```mermaid
---
config:
  theme: default
  themeVariables:
    fontSize: 18px
---
flowchart TD
    A["OSF API harvest<br/>(all OSF-hosted preprint servers)"]
    A --> B{"Ingest eligible?<br/>(date window, not post-print,<br/>PDF/DOCX available)"}
    B -- No --> X1(["Excluded:<br/>outside window / post-print /<br/>no file or unsupported format"])
    B -- Yes --> E["GROBID: PDF → TEI XML"]

    E --> F{"Content eligible?<br/>(≥ 5 pages, ≥ 1,000 words,<br/>reference section parseable)"}
    F -- No --> X2(["Excluded:<br/>content criteria"])
    F -- Yes --> H["Extract & augment<br/>reference DOIs<br/>(multi-source pipeline)"]

    H --> I{"≥ 1 cited original in FLoRA<br/>with unreferenced replication? *"}
    I -- No --> X3(["Excluded:<br/>no FLoRA match"])
    I -- Yes --> J["Extract contact emails<br/>(TEI / PDF + ORCID fallback)"]
    J --> K{"≥ 1 valid<br/>email?"}
    K -- No --> X4(["Excluded:<br/>no contact email"])
    K -- Yes --> L["Eligible preprint<br/>(baseline snapshot taken)"]

    style X1 fill:#f9d6d6,stroke:#c44,color:#900
    style X2 fill:#f9d6d6,stroke:#c44,color:#900
    style X3 fill:#f9d6d6,stroke:#c44,color:#900
    style X4 fill:#f9d6d6,stroke:#c44,color:#900
    style L fill:#d4efdf,stroke:#1e8449
```

\* FLoRA is updated regularly; preprints that initially have no FLoRA match are re-checked monthly as FLoRA expands, and enter the pipeline if a match is found.

## Part 2 — Randomisation, Intervention & Follow-up

```mermaid
---
config:
  theme: default
  themeVariables:
    fontSize: 18px
---
flowchart TD
    L["Eligible preprint<br/>(baseline snapshot taken)"] --> M["Build / update author<br/>co-authorship graph"]

    M --> N{"Connected to<br/>existing cluster?"}
    N -- "Joins BOTH treatment<br/>& control clusters" --> X2(["Excluded<br/>(cross-arm conflict;<br/>prospective only)"])
    N -- "Unconnected or<br/>single-arm cluster" --> O["Assign cluster<br/>(stratified sequential<br/>balancing by server)"]
    O --> P{Allocation}

    P -- Treatment --> Q["Send notification email<br/>(flagged originals + replications)"]
    P -- Control --> R["No contact"]

    Q --> Q1["Track bounce<br/>& survey feedback"]

    Q1 --> S["Follow-up (6 months after trial end):<br/>retrieve latest version or version of record,<br/>re-extract & augment reference list"]
    R --> S

    S --> V{"Any baseline-flagged<br/>original improved?<br/>(replication cited<br/>or original dropped)"}

    V -- Yes --> W1["Primary outcome:<br/>Improved ✓"]
    V -- No --> W2["Primary outcome:<br/>Not improved ✗"]

    style L fill:#d4efdf,stroke:#1e8449
    style X2 fill:#f9d6d6,stroke:#c44,color:#900
    style Q fill:#d6eaf8,stroke:#2980b9
    style R fill:#fdebd0,stroke:#e67e22
    style W1 fill:#d5f5e3,stroke:#27ae60
    style W2 fill:#fadbd8,stroke:#e74c3c
```

## Key to protocol sections

| Flowchart stage | Protocol section |
|---|---|
| OSF API harvest | §4 Setting / §6.1 OSF Harvesting |
| Ingest eligible? | §5.1 Inclusion criteria (1–4) |
| GROBID & content eligibility | §6.2 PDF-to-References / §5.1 criteria (5–6) |
| DOI augmentation | §7.1 DOI Augmentation |
| FLoRA matching | §7.3 Matching to FLoRA / §5.1 criterion 8 |
| Email extraction | §6.3 Email Retrieval / §5.1 criterion 7 |
| Author graph & randomisation | §8 Randomisation and Allocation |
| Notification email | §9 Intervention |
| Control (no email) | §10 Control Condition |
| Follow-up & outcome | §11 Outcomes and Measurement |
