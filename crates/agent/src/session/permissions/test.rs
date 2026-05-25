use super::*;

use agent_client_protocol::schema::PermissionOptionId;

fn id(s: &str) -> ToolCallId {
    ToolCallId::new(s.to_string())
}

#[tokio::test]
async fn resolve_wakes_waiter() {
    let gate = PermissionGate::new();
    let cancel = CancellationToken::new();
    let id = id("call-1");

    let waiter = {
        let gate = &gate;
        let cancel = cancel.clone();
        let id = id.clone();
        async move { gate.wait(id, cancel).await }
    };
    let resolver = async {
        tokio::task::yield_now().await;
        gate.resolve(
            &id,
            PermissionResolution::Selected {
                option_id: PermissionOptionId::new("allow_once".to_string()),
            },
        );
    };

    let (outcome, _) = tokio::join!(waiter, resolver);
    match outcome {
        PermissionResolution::Selected { option_id } => {
            assert_eq!(option_id.0.as_ref(), "allow_once");
        }
        other => panic!("unexpected outcome: {other:?}"),
    }
}

#[tokio::test]
async fn cancel_returns_cancelled() {
    let gate = PermissionGate::new();
    let cancel = CancellationToken::new();
    let id = id("call-2");

    let waiter = {
        let gate = &gate;
        let cancel = cancel.clone();
        let id = id.clone();
        async move { gate.wait(id, cancel).await }
    };
    let canceller = async {
        tokio::task::yield_now().await;
        cancel.cancel();
    };

    let (outcome, _) = tokio::join!(waiter, canceller);
    assert!(matches!(outcome, PermissionResolution::Cancelled));
}

#[tokio::test]
async fn resolve_without_waiter_is_noop() {
    let gate = PermissionGate::new();
    gate.resolve(
        &id("ghost"),
        PermissionResolution::Selected {
            option_id: PermissionOptionId::new("x".to_string()),
        },
    );
}
