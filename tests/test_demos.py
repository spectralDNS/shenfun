import subprocess
import os
import pytest

def test_demos():
    subprocess.check_output("/bin/bash rundemos.sh", shell=True,
                            cwd=os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    test_demos()
