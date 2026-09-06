import argparse
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import assert_sources, atomic_json, read_json


def execute(command, env, log, attempt_log, lock_fd):
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True, bufsize=0, pass_fds=(lock_fd,))
    interrupted = False
    with attempt_log.open("ab", buffering=0) as record:
        try:
            while True:
                block = os.read(process.stdout.fileno(), 65536)
                if not block:
                    break
                sys.stdout.buffer.write(block)
                sys.stdout.buffer.flush()
                log.write(block)
                record.write(block)
            return process.wait()
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            process.stdout.close()
            # The new session contains only this attempt and its workers.
            # No search by process name and no signals to other experiments.
            if process.poll() is None or interrupted:
                for sig, delay in ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        process.wait(timeout=delay)
                    except subprocess.TimeoutExpired:
                        continue
                    # Reap any remaining descendants in our isolated session.
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    break
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    experiment = Path(args.experiment).resolve()
    assert_sources(experiment)
    manifest = read_json(experiment / "RUN_MANIFEST.json")
    results = Path(manifest["results"])
    with (results / ".run.lock").open("a") as lock, (results / "suite.log").open("ab", buffering=0) as log:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This combined run already has a live launcher; no second copy was started") from None
        env = os.environ.copy()
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=manifest["training_gpus"],
                   PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1", WANDB_MODE="disabled",
                   OMP_NUM_THREADS=str(manifest["cpu_threads_per_rank"]), MKL_NUM_THREADS=str(manifest["cpu_threads_per_rank"]),
                   NCCL_ASYNC_ERROR_HANDLING="1", TOKENIZERS_PARALLELISM="false")
        env.pop("PYTHONWARNINGS", None)
        env.pop("CUDA_LAUNCH_BLOCKING", None)
        attempts_path = results / "ATTEMPTS.json"
        attempts = read_json(attempts_path) if attempts_path.exists() else []

        def announce(message):
            print(message, flush=True)
            log.write((message + "\n").encode())

        def run_phase(phase, distributed=False):
            if shutil.disk_usage(results).free < 5 * 1024**3:
                raise RuntimeError("Less than 5 GiB free; files preserved, no new phase started")
            identifier = time.strftime("%Y%m%d-%H%M%S") + f"_{len(attempts) + 1}"
            env["SEAM_ATTEMPT_ID"] = identifier
            env["CUDA_VISIBLE_DEVICES"] = manifest["evaluation_gpu"] if phase == "evaluate" else manifest["training_gpus"]
            command = [sys.executable]
            if distributed:
                command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2", "--max-restarts=0"]
            if phase == "publish":
                command += [str(experiment / "publish_combined.py"), "--experiment", str(experiment)]
            else:
                command += [str(experiment / "train_combined.py"), "--experiment", str(experiment), "--phase", phase]
            attempts_dir = results / "attempt_logs"
            attempts_dir.mkdir(exist_ok=True)
            target = attempts_dir / f"{identifier}_{phase}.log"
            announce(f"PHASE_START={phase} attempt={identifier}")
            atomic_json(results / "RUN_STATUS.json", {"phase": phase, "status": "running", "updated_unix": time.time(), "launcher_pid": os.getpid()})
            started = time.monotonic()
            result = 130
            try:
                result = execute(command, env, log, target, lock.fileno())
                return result
            finally:
                attempts.append({"phase": phase, "attempt": identifier, "exit_status": result,
                                 "seconds": round(time.monotonic() - started, 3), "log": str(target)})
                atomic_json(attempts_path, attempts)
                atomic_json(results / "RUN_STATUS.json", {"phase": phase, "status": "returned", "exit_status": result, "updated_unix": time.time()})
                announce(f"PHASE_END={phase} status={result}")

        announce(f"SEAM_COMBINED_RUN={results.name}\nTraining: 80 epochs, global batch 32, GPUs {manifest['training_gpus']}")
        # Fail on bad credentials before a long job. This reads Token.txt, never
        # switches the user's stored GitHub accounts or edits a working clone.
        from publish_combined import authenticate
        authenticate(Path(manifest["base"]) / "Token/Token.txt")
        train_marker = results / "combined/TRAINING_COMPLETE.json"
        for phase, marker, distributed in (
            ("probe", results / "ENVIRONMENT.json", False),
            ("smoke", results / "preflight/smoke_OK.json", True),
            ("smoke_resume", results / "preflight/smoke_resume_OK.json", True),
        ):
            if not train_marker.exists() and (phase == "probe" or not marker.exists()):
                if run_phase(phase, distributed) != 0:
                    raise RuntimeError(f"{phase} failed. Inspect attempt logs; no full training was started")
        if not train_marker.exists():
            restored = read_json(results / "preflight/smoke_resume_OK.json")
            if restored["previous_step"] < 1 or restored["global_step"] <= restored["previous_step"]:
                raise RuntimeError("Preflight checkpoint resume was not demonstrated")

        for retry in range(3):
            if train_marker.exists():
                break
            status = run_phase("train", True)
            if status == 0 and train_marker.exists():
                break
            if retry < 2:
                announce("Attempt failed. Retaining evidence; retrying the last completed epoch checkpoint in 60 seconds.")
                time.sleep(60)
        if not train_marker.exists():
            run_phase("publish")
            raise RuntimeError("Training attempts exhausted. Run the same launch command to resume; no results were deleted")
        if not (results / "combined/FINAL_METRICS.json").exists():
            if run_phase("evaluate") != 0:
                run_phase("publish")
                raise RuntimeError("Training is complete; evaluation needs retry. The same launcher skips training")
        if run_phase("publish") != 0:
            raise RuntimeError("Training/evaluation are complete. Rerun the launcher to retry publication only")
        atomic_json(results / "RUN_COMPLETE.json", {"training": True, "evaluation": True, "publication": True, "time": time.time()})
        announce("SEAM_COMBINED_COMPLETE=True\nTraining, final evaluation and remote snapshot verification succeeded.")
        from publish_combined import preserve_terminal
        terminal = Path(manifest["base"]) / "Terminal/SEAM_80_Epoch_Combined_Run" / results.name / "Terminal.txt"
        preserve_terminal(results / "suite.log", terminal)


if __name__ == "__main__":
    def request_stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGHUP, request_stop)
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Existing checkpoints and logs remain; use the same launcher to resume.", flush=True)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        sys.exit(1)
