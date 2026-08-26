# Dissertation

This directory is the Overleaf-ready MSc dissertation source for **Motion Prediction for Autonomous Driving: Controlled Architectural Ablations of SHARP**.

## Template fidelity

The document class, page geometry, Times typeface, line spacing, title and section formatting, front-matter order, header/footer rules, bibliography style, and chapter include structure are copied from `Dissertation_Template2026` and intentionally retained. Dissertation-specific work is limited to replacing template placeholder text, adding cited content, and adding figures under `images/`.

## Compile

Set `Dissertation_Template.tex` as the Overleaf main document and compile with pdfLaTeX. Overleaf should run BibTeX automatically; a manual sequence is:

```text
pdflatex Dissertation_Template.tex
bibtex Dissertation_Template
pdflatex Dissertation_Template.tex
pdflatex Dissertation_Template.tex
```

## Deliberate placeholders

The final three-run SHARP suite is still active. `chapters/technical2.tex` contains clearly labelled pending cells and reserved figure panels for:

1. official SHARP baseline;
2. QKNorm + uncertainty-aware target context + relative-geometry bias;
3. run 2 + residual temporal-agent Mamba.

No metric has been invented. Replace only those labelled placeholders after the suite's single-GPU validation artifacts are available. The page and heading format must not be changed.

The degree pathway was not present in the available project records, so the cover states `Master of Science` without inventing a pathway name.
