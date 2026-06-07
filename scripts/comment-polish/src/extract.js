import Parser from "tree-sitter";
import Rust from "tree-sitter-rust";

const parser = new Parser();
parser.setLanguage(Rust);

// A comment "marker" is the leading sigil that must be preserved verbatim:
//   //!  inner doc      ///  outer doc      //  plain line
//   /*! ... */ , /** ... */ , /* ... */  block forms
function classifyLine(text) {
  if (text.startsWith("//!")) return { kind: "line", marker: "//!" };
  if (text.startsWith("///")) return { kind: "line", marker: "///" };
  if (text.startsWith("//")) return { kind: "line", marker: "//" };
  return null;
}

function collectCommentNodes(root) {
  const out = [];
  const stack = [root];
  while (stack.length) {
    const n = stack.pop();
    if (n.type === "line_comment" || n.type === "block_comment") {
      out.push(n);
      continue; // do not descend into a comment's marker/content children
    }
    for (let i = n.childCount - 1; i >= 0; i--) stack.push(n.child(i));
  }
  out.sort((a, b) => a.startIndex - b.startIndex);
  return out;
}

// Is this comment the only thing on its line (modulo leading whitespace)?
// Standalone comments may merge into a block; trailing ones stay solo.
function isStandalone(src, node) {
  let i = node.startIndex - 1;
  while (i >= 0 && src[i] !== "\n") {
    if (src[i] !== " " && src[i] !== "\t") return false;
    i--;
  }
  return true;
}

// tree-sitter-rust includes the trailing newline in doc-comment (//! ///)
// nodes but not in plain // line comments. Normalize so a node's range never
// includes the trailing newline; the newline stays in the untouched tail and
// our (newline-free) reconstruction splices in cleanly.
function trimTrailingNewline(src, endIndex) {
  let e = endIndex;
  if (e > 0 && src[e - 1] === "\n") e--;
  if (e > 0 && src[e - 1] === "\r") e--;
  return e;
}

function lineNumberAt(lineStarts, index) {
  // binary search: greatest lineStart <= index
  let lo = 0;
  let hi = lineStarts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (lineStarts[mid] <= index) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

// Strip the marker (and the conventional single leading space) from one raw
// line-comment line, returning just the human text. Block comments are handled
// separately.
function stripLineMarker(raw, marker) {
  let body = raw.slice(marker.length);
  if (body.startsWith(" ")) body = body.slice(1);
  return body.replace(/\r?\n$/, "");
}

// Build the list of polishable units from one source file.
//
// A "unit" is either:
//   - a run of consecutive standalone line comments sharing the same marker and
//     indentation (a doc-comment paragraph or a code-explaining block), or
//   - a single trailing line comment, or
//   - a single block comment.
//
// Each unit carries enough metadata to (a) reconstruct the exact bytes after
// polishing and (b) build code context for the LLM.
export function extractUnits(src) {
  // tree-sitter's Node binding defaults to a 32 KB parse buffer; larger files
  // throw "Invalid argument". Size the buffer to the source (with headroom).
  const tree = parser.parse(src, undefined, {
    bufferSize: Math.max(1024 * 32, src.length * 2 + 1024),
  });
  const nodes = collectCommentNodes(tree.rootNode);

  const lineStarts = [0];
  for (let i = 0; i < src.length; i++) {
    if (src[i] === "\n") lineStarts.push(i + 1);
  }
  const srcLines = src.split("\n");

  const units = [];
  let i = 0;
  while (i < nodes.length) {
    const node = nodes[i];
    const raw = src.slice(node.startIndex, node.endIndex);

    if (node.type === "block_comment") {
      const bIndentStart = lineStarts[lineNumberAt(lineStarts, node.startIndex)];
      units.push({
        type: "block",
        indent: src.slice(bIndentStart, node.startIndex),
        startIndex: node.startIndex,
        endIndex: trimTrailingNewline(src, node.endIndex),
        startLine: lineNumberAt(lineStarts, node.startIndex),
        endLine: lineNumberAt(lineStarts, node.endIndex),
        raw,
      });
      i++;
      continue;
    }

    const cls = classifyLine(raw);
    if (!cls) {
      i++;
      continue;
    }

    const standalone = isStandalone(src, node);
    const indentStart = lineStarts[lineNumberAt(lineStarts, node.startIndex)];
    const indent = src.slice(indentStart, node.startIndex);

    if (!standalone) {
      // trailing comment: keep as its own unit
      units.push({
        type: "trailing",
        marker: cls.marker,
        indent,
        startIndex: node.startIndex,
        endIndex: trimTrailingNewline(src, node.endIndex),
        startLine: lineNumberAt(lineStarts, node.startIndex),
        endLine: lineNumberAt(lineStarts, node.endIndex),
        lines: [{ marker: cls.marker, text: stripLineMarker(raw, cls.marker) }],
      });
      i++;
      continue;
    }

    // greedily merge consecutive standalone line comments with identical marker
    // + indentation on contiguous lines
    const groupNodes = [node];
    const lines = [{ marker: cls.marker, text: stripLineMarker(raw, cls.marker) }];
    let lastLine = lineNumberAt(lineStarts, node.startIndex);
    let j = i + 1;
    while (j < nodes.length) {
      const nxt = nodes[j];
      if (nxt.type !== "line_comment") break;
      const nxtRaw = src.slice(nxt.startIndex, nxt.endIndex);
      const nxtCls = classifyLine(nxtRaw);
      if (!nxtCls || nxtCls.marker !== cls.marker) break;
      if (!isStandalone(src, nxt)) break;
      const nxtLine = lineNumberAt(lineStarts, nxt.startIndex);
      if (nxtLine !== lastLine + 1) break;
      const nxtIndentStart = lineStarts[nxtLine];
      const nxtIndent = src.slice(nxtIndentStart, nxt.startIndex);
      if (nxtIndent !== indent) break;
      groupNodes.push(nxt);
      lines.push({ marker: nxtCls.marker, text: stripLineMarker(nxtRaw, nxtCls.marker) });
      lastLine = nxtLine;
      j++;
    }

    const first = groupNodes[0];
    const last = groupNodes[groupNodes.length - 1];
    units.push({
      type: "block_lines",
      marker: cls.marker,
      indent,
      startIndex: first.startIndex,
      endIndex: trimTrailingNewline(src, last.endIndex),
      startLine: lineNumberAt(lineStarts, first.startIndex),
      endLine: lineNumberAt(lineStarts, last.startIndex),
      lines,
    });
    i = j;
  }

  return { units, srcLines };
}

// Pull a window of source lines around a unit, to give the LLM the code the
// comment is describing. Comment-only lines stay in; that is intentional —— the
// surrounding doc lines are part of the context.
export function codeContext(srcLines, unit, before = 6, after = 10) {
  const from = Math.max(0, unit.startLine - before);
  const to = Math.min(srcLines.length - 1, unit.endLine + after);
  return srcLines.slice(from, to + 1).join("\n");
}
