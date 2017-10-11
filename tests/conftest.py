import pytest
import os

def pytest_configure(config):
    os.environ['pytest'] = 'True'

def pytest_unconfigure(config):
    del os.environ['pytest']
