const API_URL = "https://api.deepseek.com/chat/completions";
const MODEL = "deepseek-chat";

const SYSTEM_PROMPT = `You are a senior Rust engineer editing source-code comments.

You will be given ONE comment from a Rust file together with the surrounding code
for context. The comment may be in Chinese, in broken machine-translated English,
or a mix. Rewrite it as a single, natural, idiomatic English comment that a native
English-speaking Rust engineer would write.

Hard rules:
- Output ONLY the rewritten comment text. No comment markers (//, ///, //!, /* */),
  no code fences, no explanations, no quotes around the output.
- Preserve meaning exactly. Do not invent behavior the code does not have. If the
  original is vague, stay equally vague rather than guessing specifics.
- Keep every identifier, type, path, macro, and rustdoc link verbatim, including
  backticks and intra-doc link syntax like [\`crate::foo::Bar\`]. Do not translate,
  rename, or reformat code tokens.
- Preserve rustdoc section headers (e.g. "# Errors", "# Panics", "# Safety",
  "## Heading") on their own lines exactly as written.
- Preserve the paragraph/line structure: if the input has multiple lines or list
  items ("- "), keep the same number of logical lines in the same order. Return a
  multi-line answer when the input was multi-line.
- Keep technical terms that are intentionally English (stdout, stderr, turn, wire,
  ACP, REPL, tracing, etc.) as-is.
- Be concise. Do not pad. Match the original's level of detail.`;

export async function polishComment({ commentText, context, signal }) {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error("DEEPSEEK_API_KEY not set");

  const user = `Rust code context (the comment is somewhere inside this window):
\`\`\`rust
${context}
\`\`\`

The comment to rewrite (without its // markers):
"""
${commentText}
"""

Rewrite this comment in idiomatic English. Output only the rewritten text.`;

  const body = {
    model: MODEL,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: user },
    ],
    temperature: 0,
    stream: false,
  };

  const res = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`DeepSeek HTTP ${res.status}: ${text.slice(0, 300)}`);
  }

  const json = await res.json();
  const content = json?.choices?.[0]?.message?.content;
  if (typeof content !== "string") {
    throw new Error(`DeepSeek: unexpected response shape: ${JSON.stringify(json).slice(0, 300)}`);
  }
  return content;
}

// Small retry wrapper for transient failures (429 / 5xx / network).
export async function polishWithRetry(args, { retries = 3, baseDelayMs = 1000 } = {}) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await polishComment(args);
    } catch (err) {
      lastErr = err;
      const msg = String(err?.message ?? err);
      const retriable = /HTTP (429|5\d\d)|fetch failed|network|ECONNRESET|ETIMEDOUT/i.test(msg);
      if (attempt === retries || !retriable) break;
      const delay = baseDelayMs * 2 ** attempt;
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  throw lastErr;
}
