"""Tests for LogDistributor."""

import time
import pytest
from log_distributor import LogDistributor


def noop_send(machine_id, packet):
    pass


def slow_send(machine_id, packet):
    time.sleep(1.0)


# ---------------------------------------------------------------------------
# Equal-weight distribution
# ---------------------------------------------------------------------------

def test_equal_weights_both_machines_receive_logs():
    """Two machines with equal weights must both receive logs over many trials."""
    hits = {"m0": 0, "m1": 0}

    def counting_send(machine_id, packet):
        hits[machine_id] += 1

    with LogDistributor(["m0", "m1"], [0.5, 0.5], counting_send) as dist:
        for _ in range(1000):
            dist.route("log-entry")

    assert hits["m0"] > 0, "m0 never received a log"
    assert hits["m1"] > 0, "m1 never received a log"
    # With 1000 samples and p=0.5 each, expect ~500 each; allow wide tolerance
    assert 480 <= hits["m0"] <= 520, f"m0 count {hits['m0']} far from expected 500"
    assert 480 <= hits["m1"] <= 520, f"m1 count {hits['m1']} far from expected 500"


# ---------------------------------------------------------------------------
# Proportional distribution
# ---------------------------------------------------------------------------

def test_weighted_distribution_is_proportional():
    """Four machines receive roughly the correct fraction of logs."""
    machines = ["c0", "c1", "c2", "c3"]
    weights = [0.4, 0.3, 0.1, 0.2]
    hits = {m: 0 for m in machines}

    def counting_send(machine_id, packet):
        hits[machine_id] += 1

    n = 10_000
    with LogDistributor(machines, weights, counting_send) as dist:
        for _ in range(n):
            dist.route("log")

    for machine, weight in zip(machines, weights):
        expected = weight * n
        # Allow ±5% absolute deviation
        assert abs(hits[machine] - expected) < 0.05 * n, (
            f"{machine}: got {hits[machine]}, expected ~{expected:.0f}"
        )


# ---------------------------------------------------------------------------
# Timeout fallback
# ---------------------------------------------------------------------------

def test_timeout_falls_back_to_another_machine():
    """When the first selected machine times out, the packet goes elsewhere."""
    routed_to = []

    def selective_slow_send(machine_id, packet):
        if machine_id == "slow":
            time.sleep(1.0)  # guaranteed to exceed 10ms timeout
        routed_to.append(machine_id)

    with LogDistributor(["slow", "fast"], [0.5, 0.5], selective_slow_send) as dist:
        # Run enough times that "slow" is almost certainly selected at least once
        for _ in range(20):
            dist.route("log")

    assert "fast" in routed_to, "fast machine never received a log"
    # "slow" may appear if it was never selected first, but should not be the only one
    assert routed_to.count("fast") > 0


def test_all_machines_timeout_raises():
    """RuntimeError is raised when every machine exceeds the 10ms timeout."""
    with LogDistributor(["m0"], [1.0], slow_send) as dist:
        with pytest.raises(RuntimeError, match="timed out"):
            dist.route("log")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        LogDistributor(["m0", "m1"], [0.4, 0.4], noop_send)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        LogDistributor(["m0", "m1"], [1.0], noop_send)


def test_closed_distributor_raises():
    dist = LogDistributor(["m0"], [1.0], noop_send)
    dist.close()
    with pytest.raises(RuntimeError, match="closed"):
        dist.route("log")
