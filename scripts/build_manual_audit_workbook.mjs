import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd().replaceAll("\\", "/");
const outputDir = `${root}/outputs/citation_audit_100`;
const rows = JSON.parse(await fs.readFile(`${outputDir}/audit_sample.json`, "utf8"));
let annotations = {};
try {
  annotations = JSON.parse(await fs.readFile(`${outputDir}/confirmed_annotations.json`, "utf8"));
} catch (error) {
  if (error?.code !== "ENOENT") throw error;
}
const clean = (value) => typeof value === "string"
  ? value.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g, "")
  : value;

const workbook = Workbook.create();
const audit = workbook.worksheets.add("Audit");
const codebook = workbook.worksheets.add("Protocol & Codebook");
workbook.comments.setSelf({ displayName: "User" });

const headers = [
  "audit_id", "dataset_id", "venue", "excerpt", "citation_marker", "raw_reference",
  "recorded_target_title", "resolved_title", "resolved_paper_id", "semantic_scholar_url",
  "target_paper_url", "source_paper_title", "source_paper_url", "resolution_query",
  "resolution_score", "discoverability_query", "discoverability_rank", "reviewer",
  "review_date", "passage_extraction", "citation_reference_mapping", "entity_resolution",
  "claim_source_relationship", "alternative_valid_sources", "overall_label_validity",
  "confidence", "error_category", "notes"
];

const values = rows.map((row) => {
  const annotation = annotations[row.audit_id] ?? {};
  return [
  row.audit_id, row.dataset_id, row.venue, row.excerpt, row.citation_marker, row.raw_reference,
  row.recorded_target_title, row.resolved_title, row.resolved_paper_id, row.semantic_scholar_url,
  row.target_paper_url, row.source_paper_title, row.source_paper_url, row.resolution_query,
  row.resolution_score, row.discoverability_query, row.discoverability_rank,
  annotation.reviewer ?? "", annotation.review_date ?? "", annotation.passage_extraction ?? "",
  annotation.citation_reference_mapping ?? "", annotation.entity_resolution ?? "",
  annotation.claim_source_relationship ?? "", annotation.alternative_valid_sources ?? "",
  annotation.overall_label_validity ?? "", annotation.confidence ?? "",
  annotation.error_category ?? "", annotation.notes ?? ""
  ].map(clean);
});

audit.getRange(`A1:AB${values.length + 1}`).values = [headers, ...values];
audit.tables.add(`A1:AB${values.length + 1}`, true, "CitationAuditTable").style = "TableStyleMedium2";
audit.freezePanes.freezeRows(1);
audit.freezePanes.freezeColumns(3);
audit.showGridLines = false;
audit.getRange("A1:AB1").format = {
  fill: "#17365D", font: { bold: true, color: "#FFFFFF" },
  wrapText: true, verticalAlignment: "center"
};
audit.getRange("A2:C101").format.fill = "#EAF2F8";
audit.getRange("D2:Q101").format.fill = "#F7F7F7";
audit.getRange("R2:AB101").format.fill = "#FFF8DC";
audit.getRange("A1:AB101").format.verticalAlignment = "top";
audit.getRange("D2:AB101").format.wrapText = true;
audit.getRange("A2:AB101").format.rowHeight = 88;
audit.getRange("A:A").format.columnWidth = 10;
audit.getRange("B:B").format.columnWidth = 12;
audit.getRange("C:C").format.columnWidth = 9;
audit.getRange("D:D").format.columnWidth = 48;
audit.getRange("E:E").format.columnWidth = 20;
audit.getRange("F:F").format.columnWidth = 48;
audit.getRange("G:H").format.columnWidth = 36;
audit.getRange("I:I").format.columnWidth = 28;
audit.getRange("J:M").format.columnWidth = 30;
audit.getRange("N:N").format.columnWidth = 28;
audit.getRange("O:O").format.columnWidth = 12;
audit.getRange("P:P").format.columnWidth = 28;
audit.getRange("Q:Q").format.columnWidth = 12;
audit.getRange("R:S").format.columnWidth = 14;
audit.getRange("T:V").format.columnWidth = 22;
audit.getRange("W:W").format.columnWidth = 25;
audit.getRange("X:Y").format.columnWidth = 22;
audit.getRange("Z:Z").format.columnWidth = 12;
audit.getRange("AA:AA").format.columnWidth = 25;
audit.getRange("AB:AB").format.columnWidth = 40;
audit.getRange("S2:S101").format.numberFormat = "yyyy-mm-dd";
audit.getRange("O2:O101").format.numberFormat = "0.000";
audit.getRange("Q2:Q101").format.numberFormat = "0";

for (const col of ["T", "U", "V", "X", "Y"]) {
  audit.getRange(`${col}2:${col}101`).dataValidation = {
    rule: { type: "list", values: ["Yes", "No", "Unclear"] }
  };
}
audit.getRange("W2:W101").dataValidation = {
  rule: { type: "list", values: ["Direct support", "Partial support", "Background attribution", "Related only", "No support", "Unclear"] }
};
audit.getRange("Z2:Z101").dataValidation = {
  rule: { type: "list", values: ["High", "Medium", "Low"] }
};
audit.getRange("AA2:AA101").dataValidation = {
  rule: { type: "list", values: ["None", "Passage extraction", "Citation mapping", "Entity resolution", "Support/relevance", "Ambiguous gold", "Other"] }
};

const protocolRows = [
  ["CiteAlign 100-Instance Manual Audit", ""],
  ["Purpose", "Estimate construction-label quality and characterize claim--source relationships in a reproducible, venue-balanced sample."],
  ["Sampling", "20 of 60 test items from each of ACL, CVPR, ICLR, ICML, and NeurIPS; deterministic SHA-256 ordering with seed citealign-audit-20260828."],
  ["Unit", "One citation-bearing passage and its recorded cited paper."],
  ["Recommended process", "Verify the source passage/citation mapping first, then compare the raw reference with the resolved scholarly record, then assess the passage--source relationship."],
  ["passage_extraction", "Yes: excerpt faithfully represents the cited passage and contains one intended citation slot. No: extraction/span is corrupted or incorrectly segmented. Unclear: source cannot be checked."],
  ["citation_reference_mapping", "Yes: citation marker maps to the displayed bibliography entry. No: wrong entry or mixed references. Unclear: source bibliography is unavailable/ambiguous."],
  ["entity_resolution", "Yes: raw bibliography entry and resolved target identify the same work. No: wrong scholarly record. Unclear: insufficient metadata."],
  ["claim_source_relationship", "Direct support; Partial support; Background attribution; Related only; No support; or Unclear."],
  ["alternative_valid_sources", "Yes when another paper could reasonably satisfy the masked citation without changing the passage; otherwise No or Unclear."],
  ["overall_label_validity", "Yes when extraction, mapping, and entity resolution are valid. Use the relationship and ambiguity columns as separate scientific annotations."],
  ["confidence", "High, Medium, or Low confidence in the completed judgment."],
  ["error_category", "Primary issue if overall_label_validity is No; use notes for secondary issues."],
  ["Analysis plan", "Report each rate with a Wilson 95% confidence interval; provide counts overall and by venue. If two reviewers annotate a subset, report raw agreement and Cohen's kappa for categorical fields."],
];
codebook.getRange(`A1:B${protocolRows.length}`).values = protocolRows;
codebook.showGridLines = false;
codebook.getRange("A1:B1").merge();
codebook.getRange("A1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF", size: 16 } };
codebook.getRange("A2:A14").format = { fill: "#D9EAF7", font: { bold: true }, verticalAlignment: "top" };
codebook.getRange("B2:B14").format = { wrapText: true, verticalAlignment: "top" };
codebook.getRange("A:A").format.columnWidth = 28;
codebook.getRange("B:B").format.columnWidth = 105;
codebook.getRange("A1:B14").format.borders = { preset: "inside", style: "thin", color: "#D9E2F3" };
codebook.getRange("A1:B14").format.autofitRows();
codebook.freezePanes.freezeRows(1);

await fs.mkdir(outputDir, { recursive: true });
const previewAudit = await workbook.render({ sheetName: "Audit", range: "A1:H12", scale: 1, format: "png" });
await fs.writeFile(`${outputDir}/audit_preview.png`, new Uint8Array(await previewAudit.arrayBuffer()));
const previewCodebook = await workbook.render({ sheetName: "Protocol & Codebook", range: "A1:B14", scale: 1, format: "png" });
await fs.writeFile(`${outputDir}/codebook_preview.png`, new Uint8Array(await previewCodebook.arrayBuffer()));

const check = await workbook.inspect({ kind: "table", range: "Audit!A1:AB8", include: "values,formulas", tableMaxRows: 8, tableMaxCols: 28 });
console.log(check.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "formula error scan" });
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
const primaryPath = `${outputDir}/citealign_manual_audit_100.xlsx`;
let savedPath = primaryPath;
try {
  await output.save(primaryPath);
} catch (error) {
  if (error?.code !== "EBUSY") throw error;
  savedPath = `${outputDir}/citealign_manual_audit_100_working_copy.xlsx`;
  await output.save(savedPath);
}
console.log(JSON.stringify({ output: savedPath, rows: rows.length }));
