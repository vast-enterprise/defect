use super::{EditOutcome, replace};

/// Helper: run the chain and unwrap success as `(new_content, count, strategy)`.
fn ok(content: &str, old: &str, new: &str, all: bool) -> (String, u32, &'static str) {
    replace(content, old, new, all).unwrap_or_else(|_| panic!("expected match"))
}

#[test]
fn exact_match_wins_and_reports_exact() {
    let (out, n, strat) = ok("alpha BETA gamma\n", "BETA", "delta", false);
    assert_eq!(out, "alpha delta gamma\n");
    assert_eq!(n, 1);
    assert_eq!(strat, "exact");
}

#[test]
fn exact_takes_precedence_over_fuzzy() {
    // A clean, uniquely-occurring exact substring must win at the strict level even though
    // looser levels would also match it.
    let content = "    alpha();\n    beta();\n";
    let (out, _n, strat) = ok(content, "    beta();", "bar();", false);
    assert_eq!(strat, "exact");
    assert_eq!(out, "    alpha();\nbar();\n");
}

#[test]
fn line_trimmed_matches_wrong_indentation() {
    // old_string has no indentation; file content is indented. Exact fails, line_trimmed
    // hits on the real span. Caveat: the matched span includes the leading indentation,
    // so splicing the unindented new_string drops it (surfaced via strategy != exact).
    let content = "fn main() {\n    let x = 1;\n    let y = 2;\n}\n";
    let (out, n, strat) = ok(content, "let x = 1;\nlet y = 2;", "let z = 3;", false);
    assert_eq!(strat, "line_trimmed");
    assert_eq!(n, 1);
    assert_eq!(out, "fn main() {\nlet z = 3;\n}\n");
}

#[test]
fn ambiguous_exact_is_rejected_not_auto_resolved() {
    let err = replace("x\nx\nx\n", "x", "y", false).err();
    assert!(matches!(err, Some(EditOutcome::Ambiguous(3))));
}

#[test]
fn replace_all_replaces_every_exact_occurrence() {
    let (out, n, _strat) = ok("x\nx\nx\n", "x", "y", true);
    assert_eq!(n, 3);
    assert_eq!(out, "y\ny\ny\n");
}

#[test]
fn not_found_when_nothing_matches() {
    let err = replace("alpha\n", "ZZZ", "y", false).err();
    assert!(matches!(err, Some(EditOutcome::NotFound)));
}

#[test]
fn line_trimmed_ambiguous_is_rejected() {
    // old_string is tab-indented (exact form absent from the space-indented file, so the
    // exact level can't fire), but its trimmed form matches two distinct blocks. The
    // line_trimmed level must reject this as ambiguous rather than silently edit the
    // first — this is stricter than opencode, which would edit the first hit.
    let content = "  foo\n    foo\n";
    let err = replace(content, "\tfoo", "bar", false).err();
    assert!(
        matches!(err, Some(EditOutcome::Ambiguous(_))),
        "expected Ambiguous, got {:?}",
        err.map(|e| match e {
            EditOutcome::Ambiguous(n) => format!("Ambiguous({n})"),
            EditOutcome::NotFound => "NotFound".to_string(),
        })
    );
}

#[test]
fn block_anchor_matches_drifting_middle() {
    // First and last lines anchor; the middle line differs slightly. Needs ≥3 lines.
    let content = "start\n  middle_actual\nend\n";
    let (out, _n, strat) = ok(content, "start\nmiddle_expected\nend", "REPLACED", false);
    assert_eq!(strat, "block_anchor");
    assert_eq!(out, "REPLACED\n");
}

#[test]
fn indentation_flexible_matches_uniform_shift() {
    // old_string indented 0; content indented 8. line_trimmed would also catch this, but
    // confirm the chain produces a correct unique splice regardless of which level hits.
    let content = "        a();\n        b();\n";
    let (out, n, _strat) = ok(content, "a();\nb();", "c();", false);
    assert_eq!(n, 1);
    assert_eq!(out, "c();\n");
}

#[test]
fn splice_preserves_surrounding_bytes_verbatim() {
    // The bytes before/after the matched span must be untouched, including trailing
    // content on the same logical region.
    let content = "header\n    target_line\nfooter\n";
    let (out, _n, _strat) = ok(content, "target_line", "REPLACED", false);
    assert_eq!(out, "header\n    REPLACED\nfooter\n");
}
