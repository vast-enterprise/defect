// Re-wrap polished comment text to a target column, matching how the original
// Chinese comments were hard-wrapped. rustfmt does NOT reflow comments (the
// `wrap_comments` option is nightly-only and off by default), so if we leave
// long single-line docs they ship long. We wrap here instead.
//
// Operates on an array of logical text lines (markers already stripped) and
// returns a new array. Rules:
//   - Lines inside ``` fenced code blocks pass through untouched.
//   - List items ("- " / "* " / "1. ") wrap with hanging indentation aligned
//     under the item text.
//   - A word longer than the budget is never split; it overflows its line.
//   - Blank lines are preserved.

function leadingWhitespace(s) {
  const m = s.match(/^\s*/);
  return m ? m[0] : "";
}

// Detect a list-item prefix and return its visible width for hanging indent,
// or null if the line is not a list item.
function listHang(line) {
  const m = line.match(/^(\s*(?:[-*+]|\d+\.)\s+)/);
  return m ? m[1].length : null;
}

function wrapOneLine(line, budget) {
  const trimmedRight = line.replace(/\s+$/, "");
  if (trimmedRight.length <= budget) return [trimmedRight];

  const lead = leadingWhitespace(trimmedRight);
  const hang = listHang(trimmedRight);
  const contIndent = " ".repeat(hang ?? lead.length);

  const words = trimmedRight.slice(lead.length).split(/\s+/);
  const out = [];
  let cur = lead;
  let curHasWord = false;

  for (const word of words) {
    const sep = curHasWord ? " " : "";
    if (curHasWord && (cur + sep + word).length > budget) {
      out.push(cur);
      cur = contIndent + word;
      curHasWord = true;
    } else {
      cur = cur + sep + word;
      curHasWord = true;
    }
  }
  if (curHasWord || out.length === 0) out.push(cur);
  return out;
}

export function wrapTextLines(lines, budget) {
  if (!(budget > 10)) return lines; // pathological prefix; don't wrap
  const out = [];
  let inFence = false;
  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      inFence = !inFence;
      out.push(line.replace(/\s+$/, ""));
      continue;
    }
    if (inFence || line.trim() === "") {
      out.push(line.replace(/\s+$/, ""));
      continue;
    }
    out.push(...wrapOneLine(line, budget));
  }
  return out;
}
