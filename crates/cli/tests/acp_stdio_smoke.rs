use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use agent_client_protocol::AcpAgent;
use agent_client_protocol_schema::{
    ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest, ProtocolVersion,
    SessionNotification, SessionUpdate, StopReason,
};

#[tokio::test]
async fn stdio_echo_round_trip() {
    let state_root = tempfile::tempdir().expect("state tempdir");
    let config_root = tempfile::tempdir().expect("config tempdir");
    let process_cwd = tempfile::tempdir().expect("process cwd tempdir");
    let cwd = tempfile::tempdir().expect("cwd tempdir");
    let prompt_text = "stdio echo smoke";

    let binary = PathBuf::from(env!("CARGO_BIN_EXE_defect"));
    let agent = AcpAgent::from_args([
        format!("XDG_STATE_HOME={}", state_root.path().display()),
        format!("XDG_CONFIG_HOME={}", config_root.path().display()),
        "sh".to_string(),
        "-c".to_string(),
        r#"cd "$1" && shift && exec "$@""#.to_string(),
        "defect-stdio-smoke".to_string(),
        process_cwd.path().display().to_string(),
        binary.display().to_string(),
        "--provider".to_string(),
        "echo".to_string(),
    ])
    .expect("valid defect command");

    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_for_handler = Arc::clone(&updates);

    let stop_reason = agent_client_protocol::Client
        .builder()
        .name("stdio-smoke-client")
        .on_receive_notification(
            async move |notification: SessionNotification, _cx| {
                updates_for_handler
                    .lock()
                    .expect("updates mutex")
                    .push(notification.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;

            let session = cx
                .send_request(NewSessionRequest::new(cwd.path()))
                .block_task()
                .await?;

            let response = cx
                .send_request(PromptRequest::new(
                    session.session_id,
                    vec![ContentBlock::from(prompt_text)],
                ))
                .block_task()
                .await?;

            Ok(response.stop_reason)
        })
        .await
        .expect("client connection completed");

    assert_eq!(stop_reason, StopReason::EndTurn);

    let updates = updates.lock().expect("updates mutex");
    let assistant_chunks: String = updates
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::AgentMessageChunk(chunk) => Some(&chunk.content),
            _ => None,
        })
        .filter_map(|content| match content {
            agent_client_protocol_schema::ContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();

    assert!(
        assistant_chunks.contains(prompt_text),
        "session updates should contain echo text; got {assistant_chunks:?}",
    );
}
