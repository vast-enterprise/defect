//! Sandbox and permission policy layer.
//!
//! v0 only provides policy-level decisions (read-only / auto / full + path allowlist);
//! OS-level isolation (landlock/seatbelt/seccomp) will be introduced later as a pluggable
//! backend.

#![cfg_attr(not(test), warn(clippy::indexing_slicing, clippy::unwrap_used))]
