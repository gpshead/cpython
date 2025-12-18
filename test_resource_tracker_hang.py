"""
Test cases to reproduce the resource_tracker hang during shutdown.

The hang occurs in _stop_locked() at the blocking waitpid() call (line 129):
    _, status = waitpid(self._pid, 0)

This blocks indefinitely when the resource tracker subprocess cannot exit
because other processes still hold the write end of the pipe open.

Stack trace when hung:
    multiprocessing/resource_tracker.py:129  _stop_locked (waitpid)
    multiprocessing/resource_tracker.py:99   _stop
    multiprocessing/resource_tracker.py:94   __del__

SCENARIOS THAT CAUSE THE HANG:

1. Inherited fd after fork: When a process forks after the resource tracker
   is started, the child inherits the write fd. If the parent tries to stop
   the tracker while the child still has the fd open, waitpid blocks forever.

2. __del__ during interpreter shutdown: When the interpreter shuts down,
   __del__ is called on _resource_tracker. If child processes still exist
   with inherited fds, the hang occurs.

3. Non-blocking lock acquisition failure (potential race): When __del__ is
   called with use_blocking_lock=False and the lock is held by another thread,
   the code still proceeds to call _stop_locked() which could lead to
   inconsistent state or hangs.

To run:
    python test_resource_tracker_hang.py [scenario_number]

    Without args: runs all scenarios with timeout protection
    With arg: runs specific scenario (may hang!)
"""

import os
import sys
import time
import signal
import unittest
import subprocess


class ResourceTrackerHangTests(unittest.TestCase):
    """Test cases demonstrating resource_tracker hang scenarios."""

    def test_hang_occurs_with_inherited_fd(self):
        """
        CONFIRMS THE HANG: When a child process inherits the resource
        tracker's write fd, calling _stop() in the parent hangs in waitpid().

        This test expects TimeoutExpired, which proves the hang occurs.
        """
        code = '''
import os
import sys
import time
import signal

def child_process():
    # Child holds the inherited fd open
    time.sleep(60)
    os._exit(0)

def main():
    from multiprocessing import resource_tracker
    resource_tracker.ensure_running()

    tracker = resource_tracker._resource_tracker

    pid = os.fork()
    if pid == 0:
        child_process()
    else:
        # This will hang because child has the fd
        tracker._stop()
        # If we reach here, no hang occurred
        os.kill(pid, signal.SIGTERM)
        os.waitpid(pid, 0)
        sys.exit(0)

if __name__ == '__main__':
    main()
'''
        # The hang is confirmed if we get TimeoutExpired
        with self.assertRaises(subprocess.TimeoutExpired,
                               msg="Expected hang (timeout) but process completed"):
            subprocess.run(
                [sys.executable, '-c', code],
                timeout=2,
                capture_output=True,
                text=True
            )

    def test_no_hang_when_child_closes_fd(self):
        """
        When the child explicitly closes the inherited fd before the
        parent calls _stop(), there should be no hang.
        """
        code = '''
import os
import sys
import time
import signal

def main():
    from multiprocessing import resource_tracker
    resource_tracker.ensure_running()

    tracker = resource_tracker._resource_tracker
    tracker_fd = tracker._fd

    pid = os.fork()
    if pid == 0:
        # Child closes the inherited fd immediately
        try:
            os.close(tracker_fd)
        except OSError:
            pass  # May already be closed
        time.sleep(2)
        os._exit(0)
    else:
        # Give child time to close the fd
        time.sleep(0.5)
        # Now _stop() should not hang
        tracker._stop()
        os.kill(pid, signal.SIGTERM)
        os.waitpid(pid, 0)
        print("SUCCESS: No hang when child closes fd")
        sys.exit(0)

if __name__ == '__main__':
    main()
'''
        result = subprocess.run(
            [sys.executable, '-c', code],
            timeout=5,
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0,
                         f"Expected success but got: stdout={result.stdout}, stderr={result.stderr}")
        self.assertIn("SUCCESS", result.stdout)

    def test_no_hang_without_fork(self):
        """Without forking, _stop() should complete normally."""
        code = '''
import sys
from multiprocessing import resource_tracker

resource_tracker.ensure_running()
tracker = resource_tracker._resource_tracker

# No fork, so only this process has the fd
tracker._stop()
print("SUCCESS: No hang without fork")
sys.exit(0)
'''
        result = subprocess.run(
            [sys.executable, '-c', code],
            timeout=5,
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("SUCCESS", result.stdout)

    def test_hang_during_interpreter_shutdown(self):
        """
        Test that the hang can occur during normal interpreter shutdown
        when __del__ is called on the ResourceTracker.

        This is the real-world scenario: a program uses multiprocessing,
        forks some children, then exits. The __del__ finalizer tries to
        stop the resource tracker, but children still have the fd.
        """
        code = '''
import os
import sys
import time
import atexit

def main():
    from multiprocessing import resource_tracker
    resource_tracker.ensure_running()

    tracker = resource_tracker._resource_tracker
    tracker_fd = tracker._fd
    tracker_pid = tracker._pid

    pid = os.fork()
    if pid == 0:
        # Child - holds fd open
        time.sleep(60)
        os._exit(0)
    else:
        # Parent - register atexit to call _stop
        # This simulates what __del__ does during shutdown
        def cleanup():
            print("atexit: calling _stop()...", flush=True)
            tracker._stop()
            print("atexit: _stop() returned", flush=True)

        atexit.register(cleanup)

        # Exit normally - atexit will call _stop which will hang
        print(f"Parent exiting, child={pid} holds fd={tracker_fd}", flush=True)
        sys.exit(0)

if __name__ == '__main__':
    main()
'''
        # This should hang (timeout) because the atexit handler calls _stop
        # while the child still has the fd open
        with self.assertRaises(subprocess.TimeoutExpired,
                               msg="Expected hang during shutdown"):
            subprocess.run(
                [sys.executable, '-c', code],
                timeout=2,
                capture_output=True,
                text=True
            )


def run_scenario_1():
    """Run scenario 1 directly (will hang without timeout protection)."""
    from multiprocessing import resource_tracker
    resource_tracker.ensure_running()

    tracker = resource_tracker._resource_tracker
    tracker_fd = tracker._fd
    tracker_pid = tracker._pid

    print(f"Resource tracker fd: {tracker_fd}", flush=True)
    print(f"Resource tracker subprocess pid: {tracker_pid}", flush=True)

    pid = os.fork()
    if pid == 0:
        print(f"[Child {os.getpid()}] Holding fd open...", flush=True)
        time.sleep(60)
        os._exit(0)
    else:
        print(f"[Parent] Forked child: {pid}", flush=True)
        print(f"[Parent] Calling _stop() - THIS WILL HANG...", flush=True)
        sys.stdout.flush()

        # This will hang!
        tracker._stop()

        print(f"[Parent] _stop() returned (unexpected)", flush=True)
        os.kill(pid, signal.SIGTERM)
        os.waitpid(pid, 0)


def run_scenario_2():
    """
    Scenario 2: Non-blocking lock acquisition in __del__.

    When __del__ is called during interpreter shutdown with
    use_blocking_lock=False, if the lock is held by another thread,
    the code still proceeds to call _stop_locked() without the lock.

    This can lead to race conditions or inconsistent state.
    """
    import threading
    from multiprocessing import resource_tracker

    resource_tracker.ensure_running()
    tracker = resource_tracker._resource_tracker

    # Simulate lock being held by another thread
    lock_held = threading.Event()
    done = threading.Event()

    def hold_lock():
        with tracker._lock:
            print("[Thread] Holding lock...", flush=True)
            lock_held.set()
            done.wait(timeout=10)
            print("[Thread] Releasing lock", flush=True)

    # Start thread that holds the lock
    t = threading.Thread(target=hold_lock)
    t.start()
    lock_held.wait()

    # Fork a child to create the hang condition
    pid = os.fork()
    if pid == 0:
        time.sleep(60)
        os._exit(0)

    print("[Parent] Calling __del__ (which uses non-blocking lock)...", flush=True)
    sys.stdout.flush()

    # This simulates what happens during interpreter shutdown
    # __del__ calls _stop(use_blocking_lock=False)
    # If lock acquisition fails, it still calls _stop_locked()
    tracker.__del__()

    print("[Parent] __del__ returned", flush=True)
    done.set()
    t.join()
    os.kill(pid, signal.SIGTERM)
    os.waitpid(pid, 0)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('1', '2'):
        scenario = sys.argv[1]
        if scenario == '1':
            run_scenario_1()
        elif scenario == '2':
            run_scenario_2()
    else:
        # Run as unittest (passes any args like -v to unittest)
        unittest.main()
