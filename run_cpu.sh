#!/usr/bin/env bash

if [[ "$OSTYPE" == "msys" ]]; then
    VENV_BIN=.venv/Scripts
else
    VENV_BIN=.venv/bin
fi

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(dirname $($VENV_BIN/python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))/lib"
source $VENV_BIN/activate
echo "YO... LD LIBRARY: $LD_LIBRARY_PATH"
python run.py --execution-providers cpu
