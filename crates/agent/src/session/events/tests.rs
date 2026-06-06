use super::*;
use crate::event::AgentEvent;

#[tokio::test]
async fn emits_to_all_subscribers() {
    let bus = EventEmitter::new();
    let mut a = bus.subscribe();
    let mut b = bus.subscribe();

    bus.emit(AgentEvent::TurnStarted).await;

    let ea = a.next().await.expect("subscriber a closed");
    let eb = b.next().await.expect("subscriber b closed");
    assert!(matches!(ea, AgentEvent::TurnStarted));
    assert!(matches!(eb, AgentEvent::TurnStarted));
}

#[tokio::test]
async fn slow_consumer_backpressures_emit() {
    // capacity = 1：填满后下次 emit 必须阻塞到消费者读取
    let bus = EventEmitter::with_capacity(1);
    let mut sub = bus.subscribe();

    bus.emit(AgentEvent::TurnStarted).await; // 填满
    let emit_fut = bus.emit(AgentEvent::TurnStarted);
    tokio::pin!(emit_fut);

    // 不消费时 emit 应当 pending
    tokio::select! {
        biased;
        () = &mut emit_fut => panic!("emit must block when channel full"),
        () = tokio::task::yield_now() => {}
    }

    // 一旦消费，emit 完成
    let _ = sub.next().await;
    emit_fut.await;
}

#[tokio::test]
async fn dropped_subscriber_is_pruned() {
    let bus = EventEmitter::new();
    {
        let _sub = bus.subscribe();
    } // dropped here
    let mut alive = bus.subscribe();

    bus.emit(AgentEvent::TurnStarted).await;
    let count = bus.senders.lock().expect("mutex poisoned").len();
    // 第一次 emit 后死链已被清掉，只剩 alive
    assert_eq!(count, 1);
    let _ = alive.next().await;
}
