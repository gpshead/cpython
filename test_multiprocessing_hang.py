"""
Test case to reproduce resource_tracker hang using normal multiprocessing APIs.

This tests whether the hang can occur through standard multiprocessing usage
(not just raw os.fork()).
"""

import os
import sys
import time
import signal
import unittest
import subprocess
import multiprocessing


class MultiprocessingHangTests(unittest.TestCase):
    """Test hang scenarios using standard multiprocessing APIs."""

    def test_hang_with_pool_and_fork(self):
        """
        Test hang when using multiprocessing.Pool with 'fork' start method.

        When the parent process exits while pool workers are still alive,
        the __del__ finalizer may hang waiting for the resource tracker.
        """
        code = '''
import sys
import time
import multiprocessing

def worker(x):
    # Worker that takes a long time
    time.sleep(60)
    return x * 2

def main():
    # Force 'fork' start method
    multiprocessing.set_start_method('fork', force=True)

    # Start resource tracker by using shared memory or just ensure_running
    from multiprocessing import resource_tracker
    resource_tracker.ensure_running()

    # Create pool - workers inherit the resource tracker fd
    pool = multiprocessing.Pool(processes=2)

    # Submit work but don't wait for it
    result = pool.apply_async(worker, (42,))

    # Give workers time to start
    time.sleep(0.5)

    # Exit without joining the pool - workers still have the fd
    # This simulates a crash or early exit scenario
    print("Exiting with workers still running...", flush=True)

    # Explicitly call _stop to trigger the hang (simulates __del__ during shutdown)
    resource_tracker._resource_tracker._stop()

    print("_stop() returned - NO HANG", flush=True)
    pool.terminate()
    pool.join()

if __name__ == '__main__':
    main()
'''
        with self.assertRaises(subprocess.TimeoutExpired,
                               msg="Expected hang with Pool workers holding fd"):
            subprocess.run(
                [sys.executable, '-c', code],
                timeout=3,
                capture_output=True,
                text=True
            )

    def test_hang_with_process_and_fork(self):
        """
        Test hang when using multiprocessing.Process with 'fork' start method.
        """
        code = '''
import sys
import time
import multiprocessing

def long_running_worker():
    # Worker that runs for a long time
    time.sleep(60)

def main():
    multiprocessing.set_start_method('fork', force=True)

    from multiprocessing import resource_tracker
    resource_tracker.ensure_running()

    # Start a Process - it inherits the resource tracker fd
    p = multiprocessing.Process(target=long_running_worker)
    p.start()

    # Give process time to start
    time.sleep(0.3)

    print(f"Started worker process {p.pid}", flush=True)

    # Try to stop the resource tracker while worker has the fd
    resource_tracker._resource_tracker._stop()

    print("_stop() returned - NO HANG", flush=True)
    p.terminate()
    p.join()

if __name__ == '__main__':
    main()
'''
        with self.assertRaises(subprocess.TimeoutExpired,
                               msg="Expected hang with Process worker holding fd"):
            subprocess.run(
                [sys.executable, '-c', code],
                timeout=3,
                capture_output=True,
                text=True
            )

    def test_hang_with_concurrent_futures_and_fork(self):
        """
        Test hang when using concurrent.futures.ProcessPoolExecutor with 'fork'.
        """
        code = '''
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def slow_task(x):
    time.sleep(60)
    return x

def main():
    multiprocessing.set_start_method('fork', force=True)

    from multiprocessing import resource_tracker
    resource_tracker.ensure_running()

    # Create executor - workers inherit resource tracker fd
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Submit work
        future = executor.submit(slow_task, 42)

        # Give worker time to start
        time.sleep(0.3)

        print("Workers started, attempting to stop resource tracker...", flush=True)

        # This will hang because workers have the fd
        resource_tracker._resource_tracker._stop()

        print("_stop() returned - NO HANG", flush=True)

if __name__ == '__main__':
    main()
'''
        with self.assertRaises(subprocess.TimeoutExpired,
                               msg="Expected hang with ProcessPoolExecutor workers"):
            subprocess.run(
                [sys.executable, '-c', code],
                timeout=3,
                capture_output=True,
                text=True
            )

    def test_no_hang_with_spawn_method(self):
        """
        Test that 'spawn' start method does NOT cause the hang.

        With 'spawn', child processes don't inherit fds, so this should work.
        """
        code = '''
import sys
import time
import multiprocessing

def worker():
    time.sleep(2)

def main():
    # Use 'spawn' - no fd inheritance
    multiprocessing.set_start_method('spawn', force=True)

    from multiprocessing import resource_tracker
    resource_tracker.ensure_running()

    p = multiprocessing.Process(target=worker)
    p.start()

    time.sleep(0.3)

    # This should NOT hang because spawn doesn't inherit fds
    resource_tracker._resource_tracker._stop()

    print("SUCCESS: No hang with spawn method", flush=True)
    p.terminate()
    p.join()

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
                         f"stdout={result.stdout}, stderr={result.stderr}")
        self.assertIn("SUCCESS", result.stdout)


if __name__ == '__main__':
    unittest.main()
