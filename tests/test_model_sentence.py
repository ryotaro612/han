import unittest
import os
import torchtext.datasets as d
from . import marker


@unittest.skipUnless(marker.run_integration_tests, "Take many time.")
class TestCase(unittest.TestCase):
    def test(self):

        train, test = d.AG_NEWS()
