test-fast:
	docker compose run --rm app python3 scripts/run_tests.py --mode fast --parallel

test-all:
	docker compose run --rm app python3 scripts/run_tests.py --mode all --parallel

test-last-failed:
	pytest --last-failed || pytest -n auto 