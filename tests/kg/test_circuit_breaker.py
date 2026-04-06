import time

from src.kg.circuit_breaker import CircuitBreaker


def test_circuit_opens_after_consecutive_failures():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=5)
    assert cb.state == "closed"
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "open"
    assert cb.should_allow_request() is False


def test_circuit_resets_on_success():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=5)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert cb.state == "closed"
    assert cb.should_allow_request() is True


def test_circuit_half_opens_after_recovery_timeout():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=1)
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "open"
    time.sleep(1.1)
    assert cb.state == "half_open"
    assert cb.should_allow_request() is True


def test_circuit_closes_after_successful_probe():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=1)
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    time.sleep(1.1)
    assert cb.state == "half_open"
    cb.record_success()
    assert cb.state == "closed"
