import unittest
import os
from . import marker


@unittest.skipUnless(marker.run_integration_tests, "Take many time.")
class TestCase(unittest.TestCase):
    def test(self):

        assert False
