import multiprocessing.connection as mp_connection
import time
import unittest
from concurrent import futures

from test import support

from .util import (
    BaseTestCase, ProcessPoolForkMixin, ProcessPoolForkserverMixin,
    ProcessPoolSpawnMixin, create_executor_tests, setup_module)


class OnErrorError(Exception):
    """Raised from the body of a ``with executor`` block by the tests."""


def _sleep_and_return(value, delay=0.1):
    time.sleep(delay)
    return value


def _signal_and_block(started, delay=60):
    started.set()
    time.sleep(delay)


class _OnErrorBase:
    def _make_executor(self, on_error):
        kwargs = {"max_workers": 2, "on_error": on_error}
        if hasattr(self, "ctx"):
            kwargs["mp_context"] = self.get_context()
        return self.executor_type(**kwargs)


class OnErrorMixin(_OnErrorBase):
    def test_default_waits(self):
        executor = self._make_executor("wait")
        fs = []
        with self.assertRaises(OnErrorError):
            with executor:
                for i in range(12):
                    fs.append(executor.submit(_sleep_and_return, i))
                raise OnErrorError
        # "wait" lets every pending future run to completion.
        self.assertTrue(all(f.done() for f in fs))
        self.assertFalse(any(f.cancelled() for f in fs))

    def test_cancel_drops_pending(self):
        executor = self._make_executor("cancel")
        fs = []
        with self.assertRaises(OnErrorError):
            with executor:
                for i in range(12):
                    fs.append(executor.submit(_sleep_and_return, i))
                raise OnErrorError
        self.assertTrue(all(f.done() for f in fs))
        # Futures that had not started running yet were cancelled.
        self.assertTrue(any(f.cancelled() for f in fs))

    def test_no_exception_ignores_on_error(self):
        executor = self._make_executor("cancel")
        fs = []
        with executor:
            for i in range(12):
                fs.append(executor.submit(_sleep_and_return, i))
        # A clean exit ignores on_error and waits for everything.
        self.assertTrue(all(f.done() for f in fs))
        self.assertFalse(any(f.cancelled() for f in fs))

    def test_callable(self):
        seen = []

        def decide(executor, exc):
            seen.append((executor, exc))
            return "cancel"

        executor = self._make_executor(decide)
        fs = []
        with self.assertRaises(OnErrorError):
            with executor:
                for i in range(12):
                    fs.append(executor.submit(_sleep_and_return, i))
                raise OnErrorError
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], executor)
        self.assertIsInstance(seen[0][1], OnErrorError)
        self.assertTrue(any(f.cancelled() for f in fs))

    def test_callable_invalid_return(self):
        # A callable that returns an unsupported action raises ValueError,
        # which propagates out of __exit__ without shutting the executor down.
        executor = self._make_executor(lambda executor, exc: "bogus")
        try:
            with self.assertRaises(ValueError):
                with executor:
                    executor.submit(_sleep_and_return, 1)
                    raise OnErrorError
        finally:
            executor.shutdown()


class ProcessOnErrorMixin(_OnErrorBase):
    def test_terminate(self):
        self._check_force("terminate")

    def test_kill(self):
        self._check_force("kill")

    def _check_force(self, action):
        started = [self.manager.Event() for _ in range(2)]
        executor = self._make_executor(action)
        with self.assertRaises(OnErrorError):
            with executor:
                for ev in started:
                    executor.submit(_signal_and_block, ev)
                for ev in started:
                    self.assertTrue(ev.wait(support.SHORT_TIMEOUT))
                # Keep the Process objects alive so their sentinels stay
                # valid; the executor drops its own references on shutdown.
                procs = list(executor._processes.values())
                self.assertTrue(procs)
                raise OnErrorError
        # Every worker should have been stopped.  Wait on the process
        # sentinels rather than is_alive()/join(), which would race with the
        # executor's manager thread reaping the same Process objects.
        sentinels = [p.sentinel for p in procs]
        deadline = time.monotonic() + support.SHORT_TIMEOUT
        while sentinels:
            timeout = deadline - time.monotonic()
            self.assertGreater(timeout, 0, "workers were not stopped in time")
            for ready in mp_connection.wait(sentinels, timeout):
                sentinels.remove(ready)

    def test_constructor_accepts_force_actions(self):
        for action in ("kill", "terminate"):
            executor = self._make_executor(action)
            executor.shutdown()


class OnErrorConstructorTest(BaseTestCase):
    def test_thread_pool_default(self):
        with futures.ThreadPoolExecutor(1) as executor:
            self.assertEqual(executor._on_error, "wait")

    def test_invalid_string(self):
        with self.assertRaises(ValueError):
            futures.ThreadPoolExecutor(1, on_error="nope")

    def test_thread_pools_reject_force_actions(self):
        for cls in (futures.ThreadPoolExecutor,
                    futures.InterpreterPoolExecutor):
            for action in ("kill", "terminate"):
                with self.assertRaises(ValueError):
                    cls(1, on_error=action)


create_executor_tests(globals(), OnErrorMixin)
create_executor_tests(globals(), ProcessOnErrorMixin,
                      executor_mixins=(ProcessPoolForkMixin,
                                       ProcessPoolForkserverMixin,
                                       ProcessPoolSpawnMixin))


def setUpModule():
    setup_module()


if __name__ == "__main__":
    unittest.main()
