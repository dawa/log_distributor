"""
log_distributor.py - High-throughput weighted log packet router.
"""

import bisect
import concurrent.futures
import random
from typing import Any, Callable


class LogDistributor:
    """Routes log packets to collector machines proportional to their weights.

    Machine selection uses random.random() + bisect_left on a precomputed
    cumulative-weights array for O(log n) routing. Each attempt has a 10ms
    timeout; on timeout the machine is excluded and the next candidate is drawn
    from the renormalized remaining distribution.

    Thread-safe: multiple threads may call route() concurrently without locks.
    """

    TIMEOUT_S: float = 0.010  # 10 milliseconds

    def __init__(
        self,
        machines: list[str],
        weights: list[float],
        send_fn: Callable[[str, Any], None],
        max_workers: int = 100,
    ) -> None:
        """
        Args:
            machines:    Ordered list of collector machine identifiers.
            weights:     Relative weights, one per machine; must sum to 1.0.
            send_fn:     Callable(machine_id, packet) that delivers the packet.
                         Injected so the distributor stays free of network logic.
            max_workers: Thread-pool size (default matches capacity=100).
        """
        if len(machines) == 0:
            raise ValueError("machines list must not be empty")
        if len(machines) != len(weights):
            raise ValueError(
                f"machines ({len(machines)}) and weights ({len(weights)}) "
                "must have the same length"
            )
        if any(w <= 0 for w in weights):
            raise ValueError("all weights must be positive")
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1.0, got {weight_sum}")

        self._machines: list[str] = list(machines)
        self._weights: list[float] = list(weights)
        self._cumulative: list[float] = self._build_cumulative(self._weights)
        self._send_fn = send_fn
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._closed = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cumulative(weights: list[float]) -> list[float]:
        """Build prefix-sum array from a list of weights. O(n)."""
        cumulative: list[float] = []
        running = 0.0
        for w in weights:
            running += w
            cumulative.append(running)
        return cumulative

    @staticmethod
    def _select_index(cumulative: list[float]) -> int:
        """Pick a bucket index proportional to weights via binary search. O(log n).

        Generates a uniform random float r in [0, 1) then finds the leftmost
        bucket whose upper boundary exceeds r.  Machine i is selected when
        r falls in [cumulative[i-1], cumulative[i]).
        """
        r = random.random()
        idx = bisect.bisect_left(cumulative, r)
        # Clamp: guards against float accumulation making cumulative[-1] slightly > 1.0
        return min(idx, len(cumulative) - 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, packet: Any) -> str:
        """Route a log packet to a collector machine.

        On the first attempt uses the precomputed cumulative array (fast path,
        zero extra allocation).  If send_fn does not return within 10ms that
        machine is excluded and the next attempt draws from the renormalized
        distribution of the remaining machines (slow path).

        Args:
            packet: The log packet to route (opaque to the distributor).

        Returns:
            The machine id that successfully accepted the packet.

        Raises:
            RuntimeError: If every machine times out or the distributor is closed.
        """
        if self._closed:
            raise RuntimeError("LogDistributor is closed")

        # Fast path: reference the precomputed cumulative — no allocation
        current_machines: list[str] = self._machines
        current_cumulative: list[float] = self._cumulative
        excluded: set[int] = set()  # original indices of timed-out machines

        while True:
            if not current_machines:
                raise RuntimeError(
                    f"All {len(self._machines)} machine(s) timed out; "
                    "packet could not be routed."
                )

            # O(log n) machine selection
            local_idx = self._select_index(current_cumulative)
            machine_id = current_machines[local_idx]

            future = self._executor.submit(self._send_fn, machine_id, packet)
            try:
                future.result(timeout=self.TIMEOUT_S)
                return machine_id
            except concurrent.futures.TimeoutError:
                # Map local index back to original to track excluded machines
                original_idx = self._machines.index(machine_id)
                excluded.add(original_idx)

                # Rebuild distribution from original weights to avoid compounding
                # float errors across multiple exclusions.
                remaining = [
                    (i, self._weights[i])
                    for i in range(len(self._machines))
                    if i not in excluded
                ]
                if not remaining:
                    raise RuntimeError(
                        f"All {len(self._machines)} machine(s) timed out; "
                        "packet could not be routed."
                    )

                rem_indices, rem_weights_tuple = zip(*remaining)
                rem_weights: list[float] = list(rem_weights_tuple)
                total = sum(rem_weights)
                normed = [w / total for w in rem_weights]

                # Slow path: freshly built local cumulative for remaining machines
                current_machines = [self._machines[i] for i in rem_indices]
                current_cumulative = self._build_cumulative(normed)

    def close(self) -> None:
        """Shut down the executor, waiting for any in-flight sends to finish."""
        self._closed = True
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "LogDistributor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
