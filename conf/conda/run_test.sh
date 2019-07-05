#!bin/bash

export OMPI_MCA_plm=isolated
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_rmaps_base_oversubscribe=yes

$PYTHON -m pip install -r "${RECIPE_DIR}/component-requirements.txt"

py.test tests/
