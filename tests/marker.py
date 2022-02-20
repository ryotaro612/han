"""Provide the flag that run integration tests."""
import os


run_integration_tests: bool = os.getenv("HAN_TEST_ALL") is not None

skip_reason = "Take much time."
