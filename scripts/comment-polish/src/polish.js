#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import { argv, exit } from "node:process";
import { extractUnits, codeContext } from "./extract.js";
import { polishWithRetry } from "./deepseek.js";
import { reconstruct } from "./reconstruct.js";

function parseArgs(args) {
  const opts = { files: [], dryRun: false, concurrency: 6, limit: Infinity, verbose: false, maxWidth: 90 };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--dry-run") opts.dryRun = true;
    else if (a === "--verbose" || a === "-v") opts.verbose = true;
    else if (a === "--concurrency") opts.concurrency = Number(args[++i]);
    else if (a === "--limit") opts.limit = Number(args[++i]);
    else if (a === "--max-width") opts.maxWidth = Number(args[++i]);
    else if (a.startsWith("--")) throw new Error(`unknown flag: ${a}`);
    else opts.files.push(a);
  }
  if (!opts.files.length) throw new Error("usage: polish.js [--dry-run] [--verbose] [--concurrency N] [--limit N] <file.rs ...>");
  return opts;
}

// A unit is worth polishing if it carries real prose. Skip empty/marker-only
// runs and lines that are pure code (e.g. `// let x = 1;` debris) — those add
// cost and risk with no benefit.
function worthPolishing(unit) {
  const text = (unit.type === "block" ? unit.raw : unit.lines.map((l) => l.text).join("\n")).trim();
  if (!text) return false;
  // Must contain at least one letter (any script, incl. CJK) or digit. `\p{L}`
  // matches CJK, so a pure-`---` divider is skipped but `思考链` is kept.
  // NOTE: do not use `\W` here — in JS `\w` is ASCII-only, so `\W` matches CJK
  // and would wrongly drop Chinese-only comments.
  if (!/[\p{L}\p{N}]/u.test(text)) return false;
  return true;
}

async function mapLimit(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await fn(items[i], i);
    }
  });
  await Promise.all(workers);
  return results;
}

async function processFile(file, opts) {
  const src = await readFile(file, "utf8");
  const { units, srcLines } = extractUnits(src);
  const targets = units.filter(worthPolishing).slice(0, opts.limit);

  if (!targets.length) {
    console.log(`  ${file}: no polishable comments`);
    return;
  }
  console.log(`  ${file}: ${targets.length} comment unit(s)`);

  const polished = await mapLimit(targets, opts.concurrency, async (unit) => {
    const commentText = unit.type === "block" ? unit.raw : unit.lines.map((l) => l.text).join("\n");
    const context = codeContext(srcLines, unit);
    try {
      const out = await polishWithRetry({ commentText, context });
      const replacement = reconstruct(unit, out, opts.maxWidth);
      if (opts.verbose) {
        console.log(`\n--- L${unit.startLine + 1} ---`);
        console.log(`BEFORE:\n${src.slice(unit.startIndex, unit.endIndex)}`);
        console.log(`AFTER:\n${replacement}`);
      }
      return { unit, replacement };
    } catch (err) {
      console.error(`    ! L${unit.startLine + 1} failed: ${err.message}`);
      return { unit, replacement: null };
    }
  });

  // Splice back-to-front so earlier byte offsets stay valid.
  const edits = polished.filter((p) => p.replacement !== null).sort((a, b) => b.unit.startIndex - a.unit.startIndex);
  let out = src;
  for (const { unit, replacement } of edits) {
    out = out.slice(0, unit.startIndex) + replacement + out.slice(unit.endIndex);
  }

  if (opts.dryRun) {
    console.log(`  ${file}: dry-run, ${edits.length} edit(s) NOT written`);
  } else {
    await writeFile(file, out, "utf8");
    console.log(`  ${file}: wrote ${edits.length} edit(s)`);
  }
}

async function main() {
  const opts = parseArgs(argv.slice(2));
  for (const file of opts.files) {
    await processFile(file, opts);
  }
}

main().catch((err) => {
  console.error(err);
  exit(1);
});
