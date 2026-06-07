use super::*;

#[test]
fn expand_no_braces() {
    assert_eq!(expand_braces("**/*.rs"), vec!["**/*.rs"]);
}

#[test]
fn expand_simple() {
    let out = expand_braces("src/foo.{ts,tsx}");
    assert_eq!(out, vec!["src/foo.ts", "src/foo.tsx"]);
}

#[test]
fn expand_three_way() {
    let out = expand_braces("**/*.{rs,toml,md}");
    assert_eq!(out, vec!["**/*.rs", "**/*.toml", "**/*.md"]);
}

#[test]
fn build_set_ok() {
    let set = build_globset("src/**/*.rs").unwrap();
    assert!(set.is_match("src/main.rs"));
    assert!(!set.is_match("src/main.ts"));
}

#[test]
fn build_set_invalid() {
    assert!(build_globset("[bad-glob").is_err());
}
