"""gRPC transport for the backup-replication benchmark.

Why this exists
---------------
The XML-RPC path measures DEFT's shipped backup as-is, which is valuable but
mixes two things the paper wants to separate. DEFT's `Backup` is a
`SimpleXMLRPCServer`: single-threaded, HTTP/1.0, `listen(5)`. HTTP/1.0 means the
server closes the connection after every response, so each replication call
performs a fresh TCP handshake against a five-deep accept backlog. When that
backlog overflows the SYN is dropped and the client waits out the kernel's
retransmission timer, which is why the measured latency tail is quantized at
~1 s and ~3 s instead of forming a smooth queueing distribution.

That is a real property of the shipped implementation, but it is a property of
its *transport*. Production NFV state replication runs over asynchronous,
multiplexed transports. This module provides that comparison point: the same
replication call over gRPC (HTTP/2), with one persistent channel per primary and
a server thread pool in place of a single accept loop.

Running both transports is the point. If feature-aware backup selection wins
under both, the result is a property of placement rather than of one RPC stack,
which is a far stronger claim than either transport alone can support.

The service exposes exactly the method DEFT's XML-RPC backup exposes
(`update_state(state) -> bool`), so only the delivery mechanism differs.
"""

import time
from concurrent import futures

import grpc

from grpc_backup import replication_pb2, replication_pb2_grpc

# Mirrors the XML-RPC harness: a bounded wait so a wedged peer cannot stall a
# run indefinitely. gRPC surfaces this as a DEADLINE_EXCEEDED status rather than
# a socket timeout, but the intent is the same.
CALL_TIMEOUT_S = 20.0

# The server's request-handling concurrency. DEFT's XML-RPC backup is strictly
# single-threaded, so MAX_WORKERS=1 reproduces that serialization over HTTP/2 and
# isolates the transport change from the concurrency change. The default of 8 is
# the realistic production setting; both are exposed so the two effects can be
# measured separately rather than conflated.
DEFAULT_MAX_WORKERS = 8


class BackupReplicationServicer(replication_pb2_grpc.BackupReplicationServicer):
    """Server-side replication handler.

    The body matches the XML-RPC benchmark's `_make_update_state`: accept the
    state, optionally sleep to emulate a degraded host, acknowledge. No state is
    persisted, exactly as in the XML-RPC path, so the two transports do the same
    amount of application work.
    """

    def __init__(self, extra_delay_s=0.0):
        self.extra_delay_s = extra_delay_s

    def UpdateState(self, request, context):
        if self.extra_delay_s:
            time.sleep(self.extra_delay_s)
        return replication_pb2.UpdateAck(ok=True)

    def UpdateStateStream(self, request_iterator, context):
        for _ in request_iterator:
            if self.extra_delay_s:
                time.sleep(self.extra_delay_s)
            yield replication_pb2.UpdateAck(ok=True)


def serve_backup(ip, port, extra_delay_s=0.0, max_workers=DEFAULT_MAX_WORKERS):
    """Entry point for one backup's own OS process.

    Must be a module-level function so it can be pickled for
    multiprocessing.Process, matching the XML-RPC harness. Each backup runs as a
    separate process for the same reason it does there: threads in one process
    share a GIL, which would confound the contention being measured.
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            # Keep the accept path comparable to the XML-RPC server's listen(5)
            # rather than letting gRPC's defaults silently absorb the load: the
            # question is what the transport does under contention, so the
            # concurrency limit must be stated, not inherited.
            ("grpc.so_reuseport", 0),
            ("grpc.max_concurrent_streams", 100),
        ],
    )
    replication_pb2_grpc.add_BackupReplicationServicer_to_server(
        BackupReplicationServicer(extra_delay_s), server)
    server.add_insecure_port("{}:{}".format(ip, port))
    server.start()
    server.wait_for_termination()


class GrpcBackupClient:
    """One persistent HTTP/2 channel to a backup, reused for every call.

    This is the substantive difference from the XML-RPC client. There, each call
    opened a new TCP connection whether the harness intended it or not, because
    the server closes HTTP/1.0 connections after responding. Here the channel is
    genuinely long-lived and gRPC multiplexes concurrent calls onto it as HTTP/2
    streams, so burst concurrency does not translate into connection churn.
    """

    def __init__(self, port, host="127.0.0.1", timeout_s=CALL_TIMEOUT_S):
        self.timeout_s = timeout_s
        self.channel = grpc.insecure_channel(
            "{}:{}".format(host, port),
            options=[
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_permit_without_calls", 1),
            ],
        )
        self.stub = replication_pb2_grpc.BackupReplicationStub(self.channel)

    def update_state(self, state):
        """Replicate one state update. Raises on failure, like the XML-RPC path."""
        self.stub.UpdateState(
            replication_pb2.StateUpdate(state=state), timeout=self.timeout_s)
        return True

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def wait_until_ready(port, host="127.0.0.1", timeout_s=10.0):
    """Block until the backup's channel is serving, or raise.

    The XML-RPC harness slept a fixed 0.3 s after forking servers and hoped they
    had bound. gRPC can report readiness exactly, so startup is synchronised
    rather than guessed; a slow fork no longer shows up as a connection error in
    the first few measured calls.
    """
    channel = grpc.insecure_channel("{}:{}".format(host, port))
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
    finally:
        channel.close()
