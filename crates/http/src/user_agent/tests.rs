use super::*;

#[test]
fn default_user_agent_is_valid_header_value() {
    // Should not panic or fall back to the fallback — the sha from build.rs is
    // guaranteed to be ASCII, and the version is semver ASCII.
    let v = default_user_agent();
    let s = v.to_str().expect("ascii header");
    assert!(s.starts_with("defect-http/"), "got {s}");
}

#[test]
fn build_sha_is_present() {
    // build.rs always provides `unknown` as a fallback, so the SHA must never be
    // empty.
    // This test mainly guards against accidentally removing that fallback later.
    let sha = env!("DEFECT_HTTP_GIT_SHA");
    assert!(!sha.is_empty(), "DEFECT_HTTP_GIT_SHA must always be set");
}
