test-fast:
	pytest -n auto -m "unit or fast"

test-all:
	pytest -n auto

test-last-failed:
	pytest --last-failed || pytest -n auto 