.PHONY: install test demo quick
install: ; pip install -e ".[dev]"
test: ; python -m pytest -q
demo: ; python run_demo.py
quick: ; python run_demo.py --quick
