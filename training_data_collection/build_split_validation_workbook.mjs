import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const splitDir = path.resolve("training_data_collection/splits");
const report = JSON.parse(
  await fs.readFile(path.join(splitDir, "split_validation.json"), "utf8"),
);

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const inputs = workbook.worksheets.add("Input Audit");

summary.showGridLines = false;
summary.getRange("A1:H1").merge();
summary.getRange("A1").values = [["CiteGuard Year-Split Validation"]];
summary.getRange("A1:H1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
summary.getRange("A1:H1").format.rowHeight = 30;

summary.getRange("A3:B8").values = [
  ["Policy", "Value"],
  ["Seed", report.policy.seed],
  ["Train source year", report.policy.train_source_year],
  ["Test source year", report.policy.test_source_year],
  ["Train quota per venue", report.policy.train_per_venue],
  ["Test quota per venue", report.policy.test_per_venue],
];

const venueRows = report.policy.venues.map((venue) => [
  venue,
  report.outputs.train.by_venue[venue],
  report.outputs.test.by_venue[venue],
]);
summary.getRange("D3:F8").values = [
  ["Venue", "Train rows", "Test rows"],
  ...venueRows,
];

summary.getRange("A10:C16").values = [
  ["Output check", "Train", "Test"],
  ["Rows", report.outputs.train.rows, report.outputs.test.rows],
  ["Unique excerpts", report.outputs.train.unique_excerpts, report.outputs.test.unique_excerpts],
  ["Unique source papers", report.outputs.train.unique_source_papers, report.outputs.test.unique_source_papers],
  ["Unique target papers", report.outputs.train.unique_target_papers, report.outputs.test.unique_target_papers],
  ["Max rows per source", report.outputs.train.max_rows_per_source, report.outputs.test.max_rows_per_source],
  ["Max rows per target", report.outputs.train.max_rows_per_target, report.outputs.test.max_rows_per_target],
];

summary.getRange("D10:F15").values = [
  ["Leakage check", "Count", "Status"],
  ["Exact excerpts across splits", report.leakage.between_train_and_test.exact_excerpts, null],
  ["Source titles across splits", report.leakage.between_train_and_test.source_paper_titles, null],
  ["Target titles across splits", report.leakage.between_train_and_test.target_paper_titles, null],
  ["Train overlap with benchmark", report.leakage.with_existing_benchmark.train, null],
  ["Test overlap with benchmark", report.leakage.with_existing_benchmark.test, null],
];
summary.getRange("F11").formulas = [["=IF(E11=0,\"PASS\",\"FAIL\")"]];
summary.getRange("F11:F15").fillDown();

summary.getRange("A18:D24").values = [
  ["2024 partition", "Full pool", "RL train", "Validation"],
  ["Rows", report.outputs.train.rows, report.outputs.rl_train.rows, report.outputs.validation.rows],
  ["Rows per venue", report.policy.train_per_venue, report.outputs.rl_train.by_venue.ACL, report.policy.validation_per_venue],
  ["Unique excerpts", report.outputs.train.unique_excerpts, report.outputs.rl_train.unique_excerpts, report.outputs.validation.unique_excerpts],
  ["Unique source papers", report.outputs.train.unique_source_papers, report.outputs.rl_train.unique_source_papers, report.outputs.validation.unique_source_papers],
  ["Unique target papers", report.outputs.train.unique_target_papers, report.outputs.rl_train.unique_target_papers, report.outputs.validation.unique_target_papers],
  ["Split year", report.policy.train_source_year, report.policy.train_source_year, report.policy.train_source_year],
];
summary.getRange("F18:H21").values = [
  ["RL train / validation overlap", "Count", "Status"],
  ["Exact excerpts", report.leakage.between_rl_train_and_validation.exact_excerpts, null],
  ["Source-paper titles", report.leakage.between_rl_train_and_validation.source_paper_titles, null],
  ["Target-paper titles", report.leakage.between_rl_train_and_validation.target_paper_titles, null],
];
summary.getRange("H19").formulas = [["=IF(G19=0,\"PASS\",\"FAIL\")"]];
summary.getRange("H19:H21").fillDown();

for (const range of ["A3:B3", "D3:F3", "A10:C10", "D10:F10", "A18:D18", "F18:H18"]) {
  summary.getRange(range).format = {
    fill: "#D9EAF7",
    font: { bold: true, color: "#17365D" },
    borders: { preset: "outside", style: "thin", color: "#9EADBA" },
  };
}
summary.getRange("F11:F15").conditionalFormats.add("cellIs", {
  operator: "equal",
  formula: '"PASS"',
  format: { fill: "#E2F0D9", font: { bold: true, color: "#375623" } },
});
summary.getRange("H19:H21").conditionalFormats.add("cellIs", {
  operator: "equal",
  formula: '"PASS"',
  format: { fill: "#E2F0D9", font: { bold: true, color: "#375623" } },
});
summary.freezePanes.freezeRows(1);
summary.getRange("A1:H24").format.autofitColumns();
summary.getRange("A1:H24").format.autofitRows();
summary.getRange("A:A").format.columnWidth = 27;
summary.getRange("D:D").format.columnWidth = 30;
summary.getRange("F:F").format.columnWidth = 31;

inputs.showGridLines = false;
const inputRows = report.inputs.map((entry) => [
  entry.year,
  entry.venue,
  entry.file,
  entry.raw_rows,
  entry.eligible_rows_after_dedup,
  entry.exact_duplicate_rows_removed,
  entry.audit.exists ? entry.audit.rows : 0,
  entry.audit.invalid_json_lines ?? 0,
  entry.audit.eligible_csv_ids_missing_from_audit ?? 0,
  entry.missing_columns.length,
  Object.values(entry.problems).reduce((sum, value) => sum + value, 0),
]);
inputs.getRange(`A1:K${inputRows.length + 1}`).values = [
  [
    "Year",
    "Venue",
    "Source file",
    "Raw rows",
    "Eligible rows",
    "Duplicates removed",
    "Audit rows",
    "Invalid audit JSON",
    "CSV IDs missing in audit",
    "Missing schema columns",
    "Rejected rows",
  ],
  ...inputRows,
];
inputs.getRange("A1:K1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
};
inputs.getRange("A1:K11").format.borders = {
  preset: "inside",
  style: "thin",
  color: "#D9E1F2",
};
inputs.freezePanes.freezeRows(1);
inputs.getRange("A1:K11").format.autofitColumns();
inputs.getRange("C:C").format.columnWidth = 31;

const summaryInspection = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:H24",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 8,
});
console.log(summaryInspection.ndjson);

const preview = await workbook.render({
  sheetName: "Summary",
  range: "A1:H24",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(
  path.join(splitDir, "split_validation_preview.png"),
  new Uint8Array(await preview.arrayBuffer()),
);

const auditPreview = await workbook.render({
  sheetName: "Input Audit",
  range: "A1:K11",
  scale: 1.15,
  format: "png",
});
await fs.writeFile(
  path.join(splitDir, "split_validation_audit_preview.png"),
  new Uint8Array(await auditPreview.arrayBuffer()),
);

const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(formulaErrors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(splitDir, "split_validation.xlsx"));
