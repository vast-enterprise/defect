use super::*;

#[test]
fn detect_lf_when_no_crlf() {
    assert_eq!(detect_line_ending("a\nb\nc\n"), LineEnding::Lf);
}

#[test]
fn detect_crlf_when_majority() {
    assert_eq!(detect_line_ending("a\r\nb\r\nc\r\n"), LineEnding::Crlf);
}

#[test]
fn detect_lf_when_mixed_majority_lf() {
    assert_eq!(detect_line_ending("a\nb\nc\r\n"), LineEnding::Lf);
}

#[test]
fn normalize_lf_to_crlf_handles_existing_crlf() {
    let input = "a\r\nb\nc";
    let out = normalize(input, LineEnding::Crlf);
    assert_eq!(&*out, "a\r\nb\r\nc");
}

#[test]
fn normalize_crlf_to_lf() {
    let out = normalize("a\r\nb\r\n", LineEnding::Lf);
    assert_eq!(&*out, "a\nb\n");
}

#[test]
fn slice_lines_with_offset_and_limit() {
    let text = "1\n2\n3\n4\n5\n";
    let out = slice_lines(text, Some(2), Some(2));
    assert_eq!(out, "2\n3\n");
}

#[test]
fn looks_binary_detects_null_byte() {
    assert!(looks_binary(b"hello\0world"));
    assert!(!looks_binary(b"hello world\nfoo"));
}
