import { wrapTextLines } from "./wrap.js";

// Turn the LLM's plain-text output back into exact source bytes for a unit,
// re-attaching the original markers and indentation. The returned string
// replaces src[unit.startIndex .. unit.endIndex].
//
// `maxWidth` is the target total line width; text is re-wrapped to fit after
// accounting for the `indent + marker + space` prefix. Trailing comments are
// never wrapped (splitting an inline comment onto its own lines would change
// the code layout).
export function reconstruct(unit, polished, maxWidth = 100) {
  const out = polished.replace(/\r\n/g, "\n").replace(/\s+$/, "");

  if (unit.type === "block") {
    return reconstructBlock(unit, out);
  }

  const { indent, marker } = unit;
  let textLines = out.split("\n").map((t) => t.trimEnd());

  if (unit.type !== "trailing") {
    const budget = maxWidth - (indent.length + marker.length + 1);
    textLines = wrapTextLines(textLines, budget);
  }

  // The node range starts at the marker, so the first line gets no indent
  // prefix (the untouched source before startIndex already supplies it).
  // Continuation lines must carry the indent themselves.
  return textLines
    .map((t, i) => {
      const text = t.replace(/\s+$/, "");
      const pre = i === 0 ? "" : indent;
      return text.length ? `${pre}${marker} ${text}` : `${pre}${marker}`;
    })
    .join("\n");
}

// Block comments (/* ... */, /** ... */, /*! ... */) are rare here. Preserve
// the opening sigil and closing, re-emitting the body. We keep it conservative:
// single-line block -> single line; multi-line -> ` * `-prefixed lines.
function reconstructBlock(unit, out) {
  const raw = unit.raw;
  const open = raw.startsWith("/**") ? "/**" : raw.startsWith("/*!") ? "/*!" : "/*";
  const outLines = out.split("\n");

  if (!raw.includes("\n") && outLines.length === 1) {
    return `${open} ${outLines[0].trim()} */`;
  }

  // Match the indentation of the opening line for continuation `*` lines.
  const indent = unit.indent ?? "";
  const body = outLines.map((t) => `${indent} * ${t.trimEnd()}`.trimEnd()).join("\n");
  return `${open}\n${body}\n${indent} */`;
}
