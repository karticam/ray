#!/usr/bin/env python3
"""
Repro: an IDLE node-drain loses a still-referenced plasma object, and because the
producing task used max_retries=0 it can't be reconstructed ->
ray.exceptions.ObjectReconstructionFailedError (the prodjob failure).

------------------------------------------------------------------------------------
MECHANISM (all verified in this branch's source):

  1. A task return >100 KiB is stored in the EXECUTOR's plasma and pinned there:
        pinned_objects_size_ += |obj|          (raylet/local_object_manager.cc:52)
     GetPrimaryBytes() = pinned_objects_size_   (raylet/local_object_manager.cc:685)

  2. The raylet decides "object store idle" from a CACHED flag,
        idle_time_states_[ObjectStoreMemory],
     which is refreshed ONLY at resource-sync time:
        CreateSyncMessage -> UpdateAvailableObjectStoreMemResource
        (raylet/scheduling/local_resource_manager.cc:340-348, called from :422)
     Pinning an object does NOT eagerly update this flag.

  3. HandleDrainRaylet reads that CACHED flag (IsLocalNodeIdle), NOT live GetPrimaryBytes:
        raylet/node_manager.cc:2203-2212
     So between a pin and the next sync, the node can report "idle" while it still
     holds a pinned, still-referenced primary copy -> an idle DrainNode is accepted ->
     the raylet shuts down -> the object's only copy is gone -> with max_retries=0 the
     owner can't reconstruct it -> ObjectReconstructionFailedError.

In production this stale window is ~one sync period (default 100 ms,
raylet_report_resources_period_milliseconds) and is hit only occasionally under
aggressive end-of-job draining. This script makes it DETERMINISTIC by widening the
sync period so the cached flag stays stale long enough to land a drain.

------------------------------------------------------------------------------------
A/B it runs:

  CONTROL (sync = 100 ms): the cache refreshes within ~100 ms, so the pinned object
      correctly keeps the node non-idle -> the drain is REJECTED ("no longer idle")
      -> object survives.  (Shows the guard working.)

  RACE (sync = 120 s): the cached object-store-idle flag stays stale while the object
      is pinned -> the idle drain is ACCEPTED -> node terminates -> object LOST ->
      ObjectReconstructionFailedError.  (Shows the bug.)

------------------------------------------------------------------------------------
Run on a Linux devbox with Ray built from this branch:

    python repro_idle_drain_object_loss.py            # both (control then race)
    python repro_idle_drain_object_loss.py control     # only the guard-works case
    python repro_idle_drain_object_loss.py race        # only the bug case

Each scenario prints its worker raylet log dir and saves that raylet's [karticam] lines
to karticam_<scenario>_raylet.log in the current directory for separate inspection.

(Uses ray.cluster_utils.Cluster to run head + worker raylets as separate processes on
one machine; no autoscaler is involved -- the drain is issued directly via
GcsClient.drain_node, exactly as autoscaler v2 / `ray drain-node` would.)
"""

import time

import ray
from ray._raylet import GcsClient
from ray.cluster_utils import Cluster
from ray.core.generated import autoscaler_pb2
from ray.exceptions import ObjectLostError, ObjectReconstructionFailedError

# >100 KiB (max_direct_call_object_size) so the return is stored in the worker's
# plasma (in_plasma=True) instead of being inlined into the owner's reply.
OBJECT_SIZE = 10 * 1024 * 1024  # 10 MiB
OBJECT_STORE = 256 * 1024 * 1024  # 256 MiB per node
IDLE = autoscaler_pb2.DrainNodeReason.Value("DRAIN_NODE_REASON_IDLE_TERMINATION")
NO_DEADLINE = 2**63 - 1


def dump_karticam_logs(worker, label: str) -> None:
    """Print + save the worker raylet's [karticam] lines so they can be inspected even
    after the cluster is torn down (raylet logs live under <node>/logs/raylet.{out,err})."""
    import os

    logs_dir = getattr(worker, "_logs_dir", None)
    if not logs_dir:
        return
    lines = []
    for fname in ("raylet.out", "raylet.err"):
        path = os.path.join(logs_dir, fname)
        if os.path.exists(path):
            with open(path, errors="replace") as f:
                lines += [ln.rstrip("\n") for ln in f if "[karticam]" in ln]
    out_path = os.path.abspath(
        f"karticam_{label.split(':')[0].strip().lower()}_raylet.log"
    )
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(
        f"\n[{label}] ----- worker raylet [karticam] lines "
        f"({len(lines)} lines; saved to {out_path}) -----"
    )
    for ln in lines:
        print("   " + ln)
    print(f"[{label}] ----- end [karticam] -----\n")


def run_scenario(sync_period_ms: int, label: str) -> str:
    print("\n" + "=" * 78)
    print(f"[{label}]  raylet_report_resources_period_milliseconds = {sync_period_ms}")
    print("=" * 78)

    cluster = Cluster()
    # Head node hosts the driver (the OWNER of the object). num_cpus=0 => no task runs
    # here, so the task is forced onto the worker. _system_config on the head propagates
    # cluster-wide via the GCS, so the worker raylet picks up the widened sync period.
    cluster.add_node(
        num_cpus=0,
        resources={"head": 1},
        object_store_memory=OBJECT_STORE,
        _system_config={"raylet_report_resources_period_milliseconds": sync_period_ms},
    )
    ray.init(address=cluster.address)
    worker = None
    try:
        # Worker node = the EXECUTOR; the lost object will live in its plasma.
        worker = cluster.add_node(
            num_cpus=1, resources={"worker": 1}, object_store_memory=OBJECT_STORE
        )
        cluster.wait_for_nodes()
        worker_id = worker.node_id
        print(
            f"[{label}] worker raylet log dir: {worker._logs_dir}  "
            "(grep raylet.out/.err here for [karticam])"
        )
        gcs = GcsClient(address=ray.get_runtime_context().gcs_address)

        @ray.remote(num_cpus=1, max_retries=0)  # max_retries=0 => no reconstruction
        def make_big_object():
            return b"x" * OBJECT_SIZE

        # Forced onto the worker (head has 0 CPU). Return >100 KiB => pinned in worker plasma.
        ref = make_big_object.remote()

        # Wait until the object is CREATED on the worker WITHOUT pulling it to the head
        # (fetch_local=False keeps the only copy on the worker).
        ready, _ = ray.wait([ref], num_returns=1, timeout=60, fetch_local=False)
        assert ready, "task did not finish in time"
        print(
            f"[{label}] object created & pinned on worker {worker_id[:12]}…; "
            "driver holds the ref but did NOT fetch it (only copy is on the worker)."
        )

        # Let the worker lease be returned so CPU + NODE_WORKERS go idle. After this the
        # ONLY thing that should keep the node non-idle is the pinned object (object-store axis).
        time.sleep(3)

        # Issue an idle-termination drain, retrying for a bit while CPU/lease settle.
        accepted, reason = False, ""
        for _ in range(15):
            accepted, reason = gcs.drain_node(
                worker_id, IDLE, "repro: idle drain while object pinned", NO_DEADLINE
            )
            if accepted:
                break
            time.sleep(1)

        if not accepted:
            print(f"[{label}] idle DrainNode REJECTED -> {reason!r}")
            val = ray.get(ref, timeout=30)
            print(
                f"[{label}] RESULT: object SAFE ({len(val)} bytes retrieved). "
                "The pinned object kept the node non-idle -> guard worked as intended."
            )
            return "protected"

        print(f"[{label}] idle DrainNode ACCEPTED while the object was pinned (!).")

        # Drain accepted -> worker shuts down -> the object's only copy is destroyed.
        for _ in range(30):
            if worker_id not in {n["NodeID"] for n in ray.nodes() if n["Alive"]}:
                break
            time.sleep(1)
        print(f"[{label}] worker node terminated (AUTOSCALER_DRAIN_IDLE).")

        # Owner still references the object -> tries to reconstruct -> max_retries=0 -> error.
        try:
            ray.get(ref, timeout=60)
            print(
                f"[{label}] RESULT: object still retrievable — repro did NOT trigger."
            )
            return "no_loss"
        except (ObjectReconstructionFailedError, ObjectLostError) as e:
            print(f"[{label}] RESULT: *** OBJECT LOST ***  {type(e).__name__}")
            print("    " + str(e).strip().replace("\n", "\n    "))
            return "lost"
    finally:
        if worker is not None:
            dump_karticam_logs(worker, label)
        ray.shutdown()
        cluster.shutdown()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce object loss from idle-draining a node that holds a pinned, "
            "still-referenced plasma object (max_retries=0)."
        )
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="both",
        choices=["control", "race", "both"],
        help=(
            "control = normal 100ms sync (guard holds, drain rejected, object safe); "
            "race = widened 120s sync (stale flag, drain accepted, object lost); "
            "both = run control then race (default)."
        ),
    )
    args = parser.parse_args()

    # (sync_period_ms, label) per scenario.
    scenarios = {
        "control": (100, "CONTROL: normal 100ms sync"),
        "race": (120_000, "RACE: widened sync window (120s)"),
    }
    to_run = ["control", "race"] if args.mode == "both" else [args.mode]

    results = {}
    for name in to_run:
        sync_period_ms, label = scenarios[name]
        results[name] = run_scenario(sync_period_ms, label)

    print("\n" + "#" * 78)
    print("# SUMMARY: " + "  ".join(f"{name}={results[name]!r}" for name in to_run))
    if args.mode == "both":
        if results.get("control") == "protected" and results.get("race") == "lost":
            print(
                "# REPRO CONFIRMED: with a fresh object-store-idle signal the drain is"
            )
            print(
                "# rejected (object safe); with a stale one it is accepted and the object"
            )
            print("# is lost -> ObjectReconstructionFailedError (max_retries=0).")
        else:
            print(
                "# Repro did not produce the expected A/B; see per-scenario output above."
            )
    else:
        print(
            f"# Ran '{args.mode}' only. Inspect karticam_{args.mode}_raylet.log "
            "(saved in CWD) for the [karticam] lines."
        )
    print("#" * 78)


if __name__ == "__main__":
    main()
