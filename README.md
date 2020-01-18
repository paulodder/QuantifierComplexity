# Getting started

Install requirements see `requirements.txt` (use python3.7)

Set `PROJECT_DIR` to the path to the directory of this repo in `.env`


# Usage


## Generating expressions

In order to **generate** expressions for a given set of settings, use
`src/generate_expressions.py`

    python src/generate_expressions.py  --max_model_size 8 --max_expression_length 6 --json_setup Logical_index.json

The script will store the results under `PROJECT_DIR/results/` by default (but
this behaviour can be changed by setting `RESULTS_DIR_RELATIVE` in `.env`).
