# Getting started

Install requirements see (`requirements.txt`)
Set PROJECT\\<sub>DIR</sub> to the path to the directory of this repo in `.env`


# Usage


## Generating expressions

In order to **generate** expressions for a given set of settings, use
`src/generate_expressions.py`

    python src/generate_expressions.py  --max_model_size 9 --max_expression_length 8 --json_setup Logical_index.json

The script will store the results under $PROJECT<sub>DIR</sub>/results/ by default.
