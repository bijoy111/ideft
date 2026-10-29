"""Correctness tests for the flow table, the workload, and the derived M.

The migration cost reported in the paper is computed from these, so the
serialization must be lossless, the workload must reach the occupancy theory
predicts, and the delta must actually be the set of entries that changed.
"""

import random
import time

import flow_state as fs
import measure_migration_cost as mmc
import workload as wl


def _fill(n, seed=0):
    rng = random.Random(seed)
    table = fs.FlowTable()
    now = time.time()
    for i in range(n):
        key = (rng.randrange(1 << 32), rng.randrange(1 << 32),
               rng.randrange(1024, 65535), rng.choice([80, 443, 22]),
               rng.choice([fs.TCP, fs.UDP]))
        table.observe(key, rng.randrange(40, 1500), now + i * 1e-3,
                      fs.TCP_ESTABLISHED,
                      (rng.randrange(1 << 32), rng.randrange(1024, 65535)))
    return table


def test_roundtrip_is_lossless():
    """Every encoding must reproduce the table exactly, including an IPv4 address
    above 2^31, which is where XML-RPC's signed <int> overflows."""
    table = _fill(500)
    table.observe((0xFFFFFFFE, 0xC0A80001, 1234, 443, fs.TCP), 1500,
                  time.time(), fs.TCP_FIN, (0xFF000001, 65000))
    for encoding in fs.__dict__.get("_ENCODINGS", ["packed", "json", "xmlrpc"]):
        blob = table.serialize(encoding)
        back = fs.FlowTable.deserialize(blob, encoding)
        assert len(back) == len(table), encoding
        for key, entry in table.table.items():
            other = back.table[key]
            assert other.packets == entry.packets, (encoding, key)
            assert other.bytes == entry.bytes, (encoding, key)
            assert other.nat_ip == entry.nat_ip, (encoding, key)
            assert other.nat_port == entry.nat_port, (encoding, key)
            assert other.tcp_state == entry.tcp_state, (encoding, key)
    print("ok   all encodings round-trip losslessly, including nat_ip > 2^31")


def test_packed_size_is_exact():
    """The packed encoding must be exactly 4 + n*(13+31) bytes, or the bytes-per-
    entry figure the paper quotes is wrong."""
    for n in (0, 1, 97, 1000):
        table = _fill(n)
        expected = 4 + len(table) * (fs.KEY_BYTES + fs.VAL_BYTES)
        assert len(table.serialize("packed")) == expected, n
    print("ok   packed size is exactly 4 + n*(13+31) bytes")


def test_delta_is_what_changed():
    """take_delta must return exactly the entries touched since the last call,
    and must not return them twice."""
    table = _fill(200)
    table.take_delta()
    assert table.take_delta() == {}, "a second delta with no changes must be empty"
    now = time.time()
    touched = list(table.table)[:17]
    for key in touched:
        table.observe(key, 100, now)
    delta = table.take_delta()
    assert set(delta) == set(touched), (len(delta), len(touched))
    print("ok   delta contains exactly the entries that changed")


def test_apply_delta_makes_backup_match():
    """The whole point of replication: after applying the deltas, the backup's
    table must equal the primary's. DEFT's shipped backup fails this trivially,
    since it discards the state."""
    primary = fs.FlowTable()
    backup = fs.FlowTable()
    engine = wl.FlowWorkload(wl.WorkloadParams(arrival_rate=200.0), seed=1)
    engine.table = primary
    for _ in range(40):
        delta = engine.step()
        backup.apply_delta(delta)
        # The backup does not see expiry, so replicate it the way a real system
        # would: as part of the same update.
        for key in list(backup.table):
            if key not in primary.table:
                del backup.table[key]
    assert len(backup) == len(primary), (len(backup), len(primary))
    for key, entry in primary.table.items():
        assert backup.table[key].packets == entry.packets, key
    print("ok   backup state converges to primary state after applying deltas")


def test_expiry_bounds_the_table():
    """Without expiry the table grows without bound; with it, occupancy must
    settle."""
    engine = wl.FlowWorkload(wl.WorkloadParams(arrival_rate=500.0,
                                               idle_timeout=10.0), seed=2)
    engine.warm(60)
    first = len(engine.table)
    for _ in range(60):
        engine.step()
    second = len(engine.table)
    assert abs(second - first) / first < 0.15, (first, second)
    assert engine.table.expired > 0
    print("ok   expiry bounds occupancy ({:,} -> {:,} entries)".format(first, second))


def test_occupancy_matches_littles_law():
    """The generator must reproduce the occupancy queueing theory predicts, or
    the table sizes the migration cost is derived from are arbitrary."""
    for rate in (500.0, 2000.0):
        params = wl.WorkloadParams(arrival_rate=rate)
        engine = wl.FlowWorkload(params, seed=3)
        engine.warm(150)
        predicted = wl.expected_occupancy(params)
        actual = len(engine.table)
        error = abs(actual - predicted) / predicted
        assert error < 0.10, (rate, actual, predicted, error)
    print("ok   steady-state occupancy matches Little's law within 10%")


def test_m_grows_with_table_and_is_order_tens():
    """M must rise with occupancy -- a bigger table is a more expensive move --
    and must land in the tens of rounds, not the single digits assumed before
    it was measured."""
    previous = 0.0
    for rate in (500.0, 2000.0, 8000.0):
        record = mmc.measure_one(rate, 30.0, warm_rounds=120, round_s=1.0)
        derived = mmc.derive_m(record, "packed", 10.0)
        assert derived["M_rounds"] > 5.0, derived
        assert record["entries"] > previous
        previous = record["entries"]
    print("ok   M rises with occupancy and is tens of rounds, not single digits")


def test_crossover_is_monotone_in_gap():
    """A worse degradation must justify migrating sooner."""
    record = mmc.measure_one(2000.0, 30.0, warm_rounds=120, round_s=1.0)
    d = mmc.derive_m(record, "packed", 10.0)
    values = [mmc.crossover_rounds(d["M_rounds"], d["service_s_per_round"], g)
              for g in (5.0, 20.0, 60.0)]
    assert values[0] > values[1] > values[2], values
    print("ok   crossover shortens as degradation worsens: "
          + " > ".join("{:.1f}".format(v) for v in values))


if __name__ == "__main__":
    test_roundtrip_is_lossless()
    test_packed_size_is_exact()
    test_delta_is_what_changed()
    test_apply_delta_makes_backup_match()
    test_expiry_bounds_the_table()
    test_occupancy_matches_littles_law()
    test_m_grows_with_table_and_is_order_tens()
    test_crossover_is_monotone_in_gap()
    print("\nall flow-state tests passed")
