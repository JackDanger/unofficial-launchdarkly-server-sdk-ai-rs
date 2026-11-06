use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::future::BoxFuture;
use futures::FutureExt;

pub(crate) fn spawn_detached(future: impl Future<Output = ()> + Send + 'static) {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            handle.spawn(future);
        }
        Err(_) => {
            std::thread::spawn(move || futures::executor::block_on(future));
        }
    }
}

enum EagerState<T> {
    Spawned(tokio::task::JoinHandle<T>),
    Lazy(BoxFuture<'static, T>),
    Ready(Option<T>),
}

/// A value computed in the background as soon as it is created, like a JavaScript
/// promise: work proceeds whether or not anyone awaits it.
///
/// On a Tokio runtime the work is spawned as a task; with no runtime available it
/// runs when awaited. If the task panics, awaiting yields the fallback value.
pub struct Eager<T: Send + 'static> {
    state: EagerState<T>,
    fallback: fn() -> T,
}

impl<T: Send + 'static> Eager<T> {
    pub(crate) fn spawn(future: impl Future<Output = T> + Send + 'static, fallback: fn() -> T) -> Self {
        let state = match tokio::runtime::Handle::try_current() {
            Ok(handle) => EagerState::Spawned(handle.spawn(future)),
            Err(_) => EagerState::Lazy(future.boxed()),
        };
        Self { state, fallback }
    }

    pub(crate) fn ready(value: T, fallback: fn() -> T) -> Self {
        Self {
            state: EagerState::Ready(Some(value)),
            fallback,
        }
    }
}

impl<T: Send + 'static> Unpin for Eager<T> {}

impl<T: Send + 'static> Future for Eager<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        let this = self.get_mut();
        match &mut this.state {
            EagerState::Spawned(handle) => handle
                .poll_unpin(cx)
                .map(|joined| joined.unwrap_or_else(|_| (this.fallback)())),
            EagerState::Lazy(future) => future.poll_unpin(cx),
            EagerState::Ready(value) => {
                Poll::Ready(value.take().unwrap_or_else(|| (this.fallback)()))
            }
        }
    }
}

impl<T: Send + 'static> std::fmt::Debug for Eager<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = match self.state {
            EagerState::Spawned(_) => "spawned",
            EagerState::Lazy(_) => "lazy",
            EagerState::Ready(_) => "ready",
        };
        f.debug_struct("Eager").field("state", &state).finish()
    }
}
