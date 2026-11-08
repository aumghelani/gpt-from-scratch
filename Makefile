.PHONY: check test lint tour demo reproduce
PY = .venv/bin

check:     ## lint + tests (fails on any error)
	./scripts/check.sh
test:
	$(PY)/pytest
lint:
	$(PY)/ruff check src tests
tour:      ## walk one prompt through the trained model
	./scripts/demo.sh tour
demo:      ## replay the demo from trained checkpoints
	./scripts/demo.sh all
reproduce: ## retrain every checkpoint from scratch (long; see README)
	./scripts/reproduce.sh
