# Part 2: checking the extracted release text

Checks that the document part 1 chose for each filing is the actual earnings release, and that its text was extracted
cleanly. This step builds no model inputs; it validates part 1's extraction.

**Notebook: `2_EDGAR.ipynb`**
1. Reads part 1's `extraction_report.csv` and draws a sample of filings, by year and document type (EX-99.1, other
   EX-99, 8-K body).
2. Re-fetches the sampled documents from EDGAR and re-extracts them. It checks the result matches part 1's, and measures
   how much text the table-removal rule drops.
3. Builds a manual hand-check sheet, adds automatic checks, and scores extraction accuracy.
4. Writes cleaned texts for the event table.

**Inputs:** part 1's `extraction_report.csv` and `event_table.csv`, and EDGAR (needs `SEC_USER_AGENT`).

**Outputs (in `results/`):**
- checks: `extraction_reproduction.csv`, `table_rule_loss.csv`, `extraction_accuracy.csv`, `extraction_failures.csv`;
- hand-check: `extraction_handcheck.csv` and `.html`;
- cleaned texts: `release_texts.csv`;
- figures: `extraction_accuracy.png`, `extraction_quality.png`.
