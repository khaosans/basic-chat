test-fast:
	docker compose run --rm app python scripts/run_tests.py --mode fast --parallel

test-all:
	docker compose run --rm app python3 scripts/run_tests.py --mode all --parallel

test-last-failed:
	pytest --last-failed || pytest -n auto 

test-long:
	docker compose run --rm app python3 -m pytest tests/test_deep_research_full.py -v --timeout=1200 