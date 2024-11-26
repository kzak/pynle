.PHONY: install
install:
	uv sync

.PHONY: check
check:
	uv run ruff check

.PHONY: clean
clean:
	rm -f uv.lock
	rm -rf .venv

.PHONY: jupyter
jupyter:
	uv run jupyter lab --ip='*' --no-browser --NotebookApp.token='' --NotebookApp.password=''
