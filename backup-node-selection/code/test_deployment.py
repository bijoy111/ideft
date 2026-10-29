"""Tests for the multi-node deployment layer.

The point of these is that the remote code path is exercised for real, not just
type-checked. `--local` placement puts the servers on this machine but drives the
whole manifest / RemoteDeployment / Endpoint path, so a genuine experiment trial
runs through the code a testbed would use. That is what catches a hardcoded
address or a dropped teardown, which is most of what can go wrong here.
"""

import json
import os
import subprocess
import sys
import time

import deployment
import provision_remote as pr


def test_plan_gives_each_server_its_own_core():
    hosts = ["w{}".format(i) for i in range(1, 17)]
    slots = pr.plan(hosts, ["n0", "n1"], per_node=8, base_port=19000)
    assert len(slots) == 16
    seen = {}
    for host_id, spec in slots.items():
        key = (spec["node"], spec["core"])
        assert key not in seen, "core shared by {} and {}".format(host_id, seen.get(key))
        seen[key] = host_id
    # Ports may repeat across nodes; they must not repeat on one node.
    per_node_ports = {}
    for spec in slots.values():
        per_node_ports.setdefault(spec["node"], []).append(spec["port"])
    for node, ports in per_node_ports.items():
        assert len(ports) == len(set(ports)), node
    print("ok   plan gives every server its own core and a free port on its node")


def test_plan_refuses_to_oversubscribe():
    try:
        pr.plan(["w{}".format(i) for i in range(1, 20)], ["n0"], per_node=8,
                base_port=19000)
    except ValueError as exc:
        assert "Add nodes" in str(exc)
        print("ok   plan refuses to place more hosts than there are slots")
        return
    raise AssertionError("expected a capacity error")


def test_reserve_cores_is_respected():
    slots = pr.plan(["w1", "w2"], ["n0"], per_node=2, base_port=19000,
                    reserve_cores=4)
    assert min(s["core"] for s in slots.values()) == 4
    print("ok   low cores are left for the OS")


def test_manifest_round_trip_and_guards():
    path = "/tmp/ideft_test_manifest.json"
    deployment.save_manifest(path, {"w1": {"ip": "10.0.0.1", "port": 19000,
                                           "node": "n0", "core": 2}},
                             transport="grpc")
    deploy = deployment.RemoteDeployment.from_file(path)
    ep = deploy.endpoints_for(["w1"])["w1"]
    assert (ep.ip, ep.port) == ("10.0.0.1", 19000)
    assert ep.url() == "http://10.0.0.1:19000"

    for bad, why in (({"version": 99, "hosts": {"w1": {}}}, "version"),
                     ({"version": 1, "hosts": {}}, "no hosts")):
        json.dump(bad, open(path, "w"))
        try:
            deployment.RemoteDeployment.from_file(path)
        except ValueError:
            pass
        else:
            raise AssertionError("expected rejection for {}".format(why))
    print("ok   manifest round-trips and bad manifests are rejected")


def test_remote_rejects_delay_it_cannot_apply():
    """A remote deployment must not silently drop a requested degradation, or a
    run would report degraded hosts that were never degraded."""
    path = "/tmp/ideft_test_manifest2.json"
    deployment.save_manifest(path, {"w1": {"ip": "127.0.0.1", "port": 19000}})
    deploy = deployment.RemoteDeployment.from_file(path)
    try:
        deploy.endpoints_for(["w1"], slow_hosts={"w1": 0.05})
    except ValueError as exc:
        assert "cannot inject" in str(exc)
        print("ok   remote deployment refuses a delay it cannot apply")
        return
    raise AssertionError("expected refusal")


def _provision_local(n_hosts, transport, manifest, base_port):
    argv = ["--local", "--hosts", str(n_hosts), "--per-node", str(n_hosts),
            "--base-port", str(base_port), "--transport", transport,
            "--manifest", manifest]
    rc = pr.main(argv)
    assert rc == 0
    return manifest


def test_real_trial_through_the_remote_path():
    """The decisive test: run an actual experiment trial against servers started
    outside the harness, reached through a manifest."""
    import rpc_contention_experiment as rpc

    manifest = "/tmp/ideft_manifest_local.json"
    _provision_local(4, "xmlrpc", manifest, base_port=19600)
    try:
        deploy = deployment.RemoteDeployment.from_file(manifest)
        endpoints = deploy.endpoints_for(["w1", "w2", "w3", "w4"])
        deploy.wait_ready(endpoints, "xmlrpc")

        mapping = {"p1": "w1", "p2": "w2", "p3": "w1"}
        records = rpc.run_trial(mapping, base_port=19600, n_calls=12,
                                burst_size=3, manifest=manifest)
        assert set(records) == set(mapping)
        for primary, rows in records.items():
            assert rows, primary
            assert all(r["succeeded"] for r in rows), primary
            assert all(r["service_ms"] > 0 for r in rows), primary
        total = sum(len(r) for r in records.values())
        assert total == 12 * len(mapping), total
        print("ok   a real trial runs through the manifest path "
              "({} calls, all succeeded)".format(total))
    finally:
        pr.main(["--manifest", manifest, "--teardown"])
        time.sleep(0.4)


def test_local_deployment_still_works():
    """Regression: the original single-machine path must be unchanged, since every
    published result was produced with it."""
    import rpc_contention_experiment as rpc

    mapping = {"p1": "w1", "p2": "w2", "p3": "w1"}
    records = rpc.run_trial(mapping, base_port=19700, n_calls=10, burst_size=2)
    assert set(records) == set(mapping)
    assert all(all(r["succeeded"] for r in rows) for rows in records.values())
    assert sum(len(r) for r in records.values()) == 10 * len(mapping)
    print("ok   local deployment path unchanged and still measuring")


def test_env_var_selects_remote():
    """An existing experiment command must pick up a testbed from the
    environment, without the command itself changing."""
    path = "/tmp/ideft_test_manifest3.json"
    deployment.save_manifest(path, {"w1": {"ip": "10.9.9.9", "port": 19000}})
    os.environ[deployment.MANIFEST_ENV] = path
    try:
        deploy = deployment.make_deployment(19000)
        assert isinstance(deploy, deployment.RemoteDeployment)
        assert deploy.endpoints_for(["w1"])["w1"].ip == "10.9.9.9"
    finally:
        del os.environ[deployment.MANIFEST_ENV]
    assert isinstance(deployment.make_deployment(19000), deployment.LocalDeployment)
    print("ok   IDEFT_MANIFEST selects the remote deployment")


def test_wait_ready_reports_a_dead_server():
    """A server that never comes up must fail loudly, not be measured as a slow
    call. The old harness slept a fixed interval and would have recorded
    connection failures instead."""
    ep = {"w1": deployment.Endpoint("w1", "127.0.0.1", 19899)}
    start = time.time()
    try:
        deployment.Deployment().wait_ready(ep, "xmlrpc")
    except RuntimeError as exc:
        assert "did not come up" in str(exc)
        print("ok   a server that never binds raises after {:.1f}s".format(
            time.time() - start))
        return
    raise AssertionError("expected a readiness failure")


if __name__ == "__main__":
    test_plan_gives_each_server_its_own_core()
    test_plan_refuses_to_oversubscribe()
    test_reserve_cores_is_respected()
    test_manifest_round_trip_and_guards()
    test_remote_rejects_delay_it_cannot_apply()
    test_env_var_selects_remote()
    test_wait_ready_reports_a_dead_server()
    test_local_deployment_still_works()
    test_real_trial_through_the_remote_path()
    print("\nall deployment tests passed")
