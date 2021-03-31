#!/bin bash

conda env create -f dedalusenv.yml
source activate dedalusCG
if [ ! -d dedalus ]
then
    git clone https://github.com/DedalusProject/dedalus.git
    cd dedalus
else
    cd dedalus
    git pull
fi
git checkout d3
python setup.py build_ext -i
export PYTHONPATH=$PYTHONPATH:$PWD
