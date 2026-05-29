//! Langfuse 批量上报器。
//!
//! 形状：`enqueue`（非阻塞）→ 有界 mpsc → 后台 flush 任务（攒满 N 条或隔 T 秒）
//! → 复用 `defect-http` 的 [`HttpStack`] POST `/api/public/ingestion`。
//!
//! ## 可丢弃降级（硬约束）
//!
//! Langfuse 是旁路遥测，**任何故障都不得影响 agent 主循环**：
//! - `enqueue` 用 `try_send`，channel 满即**丢弃并计数告警**，绝不阻塞；
//! - POST 失败只 `warn!`，**不重试**（避免堆积反压）；
//! - 207（partial success）读 body 把 errors 记进日志，但不影响后续。
//!
//! 设计详见 `docs/internal/observability-langfuse.md` §4。

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use bytes::Bytes;
use defect_http::HttpStack;
use http::header::{AUTHORIZATION, CONTENT_TYPE};
use http::{Method, Request};
use http_body_util::{BodyExt, Full};
use tokio::sync::{mpsc, oneshot};
use tower::ServiceExt;

use super::model::{IngestionBatch, IngestionEvent};

/// 后台任务的指令。
enum Cmd {
    /// 一条待上报事件。
    Event(Box<IngestionEvent>),
    /// 立即冲刷缓冲，完成后 ack（用于退出前 flush）。
    Flush(oneshot::Sender<()>),
}

/// 上报器句柄。`Clone` 廉价（内部 `Arc`）——每 session 的 observer 各持一份。
#[derive(Clone)]
pub struct LangfuseIngest {
    tx: mpsc::Sender<Cmd>,
    /// 因 channel 满而丢弃的事件累计数。仅用于节流告警。
    dropped: Arc<AtomicU64>,
}

/// 上报器构造配置。
pub struct IngestConfig {
    /// 已建好的 HTTP 栈（与 LLM provider 共用，含超时/重试/代理/UA/trace）。
    pub http: HttpStack,
    /// Langfuse host，如 `https://cloud.langfuse.com`（不带尾斜杠）。
    pub host: String,
    /// 公钥。
    pub public_key: String,
    /// 私钥。
    pub secret_key: String,
    /// 攒满多少条立即冲刷。
    pub max_batch: usize,
    /// 周期冲刷间隔。
    pub flush_interval: Duration,
    /// 入队 channel 容量（背压边界；满了丢弃）。
    pub queue_capacity: usize,
}

impl LangfuseIngest {
    /// 启动后台 flush 任务，返回句柄。
    pub fn spawn(config: IngestConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.queue_capacity);
        let dropped = Arc::new(AtomicU64::new(0));

        let auth = {
            let raw = format!("{}:{}", config.public_key, config.secret_key);
            format!("Basic {}", BASE64.encode(raw.as_bytes()))
        };
        let endpoint = format!("{}/api/public/ingestion", config.host.trim_end_matches('/'));

        let worker = Worker {
            rx,
            http: config.http,
            endpoint,
            auth,
            max_batch: config.max_batch.max(1),
            flush_interval: config.flush_interval,
        };
        tokio::spawn(worker.run());

        Self { tx, dropped }
    }

    /// 非阻塞入队。channel 满即丢弃并计数——绝不阻塞调用方（agent 主循环）。
    pub fn enqueue(&self, event: IngestionEvent) {
        if self.tx.try_send(Cmd::Event(Box::new(event))).is_err() {
            let n = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
            // 节流告警：每丢够一批才 warn 一次，避免日志风暴。
            if n.is_multiple_of(256) {
                tracing::warn!(
                    dropped_total = n,
                    "langfuse ingest queue full; dropping telemetry events (agent unaffected)"
                );
            }
        }
    }

    /// 冲刷缓冲并等待完成。用于 session 流结束 / 进程退出前尽力送达。
    ///
    /// 后台任务已退出（接收端关闭）时直接返回——尽力而为，不保证送达。
    pub async fn flush(&self) {
        let (ack_tx, ack_rx) = oneshot::channel();
        if self.tx.send(Cmd::Flush(ack_tx)).await.is_ok() {
            let _ = ack_rx.await;
        }
    }
}

/// 后台 flush 任务的状态。
struct Worker {
    rx: mpsc::Receiver<Cmd>,
    http: HttpStack,
    endpoint: String,
    auth: String,
    max_batch: usize,
    flush_interval: Duration,
}

impl Worker {
    async fn run(mut self) {
        let mut buf: Vec<IngestionEvent> = Vec::new();
        let mut tick = tokio::time::interval(self.flush_interval);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                cmd = self.rx.recv() => match cmd {
                    Some(Cmd::Event(ev)) => {
                        buf.push(*ev);
                        if buf.len() >= self.max_batch {
                            self.send_batch(&mut buf).await;
                        }
                    }
                    Some(Cmd::Flush(ack)) => {
                        self.send_batch(&mut buf).await;
                        let _ = ack.send(());
                    }
                    // 所有 sender 已 drop：冲刷残留后退出。
                    None => {
                        self.send_batch(&mut buf).await;
                        break;
                    }
                },
                _ = tick.tick() => {
                    self.send_batch(&mut buf).await;
                }
            }
        }
    }

    /// 把当前缓冲打包成一次请求发出。空缓冲是 no-op。
    async fn send_batch(&self, buf: &mut Vec<IngestionEvent>) {
        if buf.is_empty() {
            return;
        }
        let batch = std::mem::take(buf);
        let body = match serde_json::to_vec(&IngestionBatch { batch }) {
            Ok(b) => b,
            Err(err) => {
                tracing::warn!(%err, "langfuse: failed to serialize ingestion batch; dropped");
                return;
            }
        };

        let request = match Request::builder()
            .method(Method::POST)
            .uri(&self.endpoint)
            .header(AUTHORIZATION, &self.auth)
            .header(CONTENT_TYPE, "application/json")
            .body(toac::body::Body::new(Full::new(Bytes::from(body))))
        {
            Ok(req) => req,
            Err(err) => {
                tracing::warn!(%err, "langfuse: failed to build ingestion request; dropped");
                return;
            }
        };

        // HttpStack 是 Clone 的 tower service——克隆出独立副本走 oneshot。
        match self.http.clone().oneshot(request).await {
            Ok(resp) => self.inspect_response(resp).await,
            Err(err) => {
                tracing::warn!(%err, "langfuse: ingestion POST failed; batch dropped (no retry)");
            }
        }
    }

    /// 检查响应。Langfuse 输入错误不返 4xx，而是 207 带 `{errors, successes}`。
    async fn inspect_response(&self, resp: http::Response<hyper::body::Incoming>) {
        let status = resp.status();
        if status.is_success() && status.as_u16() != 207 {
            return;
        }
        // 207 或非 2xx：收集 body 用于诊断（只读一小段，避免大响应）。
        let body = match resp.into_body().collect().await {
            Ok(collected) => collected.to_bytes(),
            Err(err) => {
                tracing::warn!(%status, %err, "langfuse: ingestion non-OK; body unreadable");
                return;
            }
        };
        let snippet = String::from_utf8_lossy(&body);
        let snippet = snippet.chars().take(1024).collect::<String>();
        tracing::warn!(%status, body = %snippet, "langfuse: ingestion returned errors");
    }
}
