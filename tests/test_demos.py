import pytest
import subprocess
import os
from numbers import Number

def test_demos():
    output = subprocess.check_output("/bin/bash rundemos.sh", shell=True,
                                     cwd=os.path.join(os.getcwd(), "tests"))

if __name__ == '__main__':
    test_demos()
