use super::*;

#[test]
fn relative_path_resolves_inside_workspace() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::write(root.join("a.txt"), "x").unwrap();

    let resolved = resolve_workspace_path(root, Path::new("a.txt")).unwrap();
    let expected = std::fs::canonicalize(root).unwrap().join("a.txt");
    assert_eq!(resolved, expected);
}

#[test]
fn missing_parent_resolves_by_walking_up() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    // Parent directory does not exist — walk up to the workspace root and rejoin the
    // missing segment.
    let resolved = resolve_workspace_path(root, Path::new("missing_dir/sub/file.txt")).unwrap();
    let root_canon = std::fs::canonicalize(root).unwrap();
    let expected = root_canon.join("missing_dir/sub/file.txt");
    assert_eq!(resolved, expected);
}

#[test]
fn parent_canonicalize_blocks_traversal() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    // `workspace_root/..` is the parent directory — after canonicalizing the parent, it
    // is guaranteed to be outside the workspace.
    let err = resolve_workspace_path(root, Path::new("../escape.txt")).unwrap_err();
    assert!(matches!(err, FsError::NotPermitted(_)), "got {err:?}");
}

#[test]
fn write_target_may_not_exist() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    // Target file does not exist but its parent directory does → should succeed
    let resolved = resolve_workspace_path(root, Path::new("new.txt")).unwrap();
    let expected = std::fs::canonicalize(root).unwrap().join("new.txt");
    assert_eq!(resolved, expected);
}

#[test]
fn symlink_pointing_outside_workspace_is_blocked() {
    #[cfg(unix)]
    {
        let dir = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let root = dir.path();
        // Symlink: workspace/escape -> /tmp/other-tempdir/
        std::os::unix::fs::symlink(other.path(), root.join("escape")).unwrap();

        let err = resolve_workspace_path(root, Path::new("escape/file.txt")).unwrap_err();
        assert!(matches!(err, FsError::NotPermitted(_)), "got {err:?}");
    }
}
