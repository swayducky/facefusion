#!/usr/bin/env bash

if [[ "$OSTYPE" == "msys" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate	
fi
python run.py --execution-providers cuda
