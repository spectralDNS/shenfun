import pytest
import subprocess
import os

def test_demos():
    subprocess.check_output("/bin/bash rundemos.sh", shell=True,
                            cwd=os.path.join(os.getcwd(), "tests"))

if __name__ == '__main__':
    test_demos()
