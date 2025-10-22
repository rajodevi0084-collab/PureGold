UV ?= uv

.PHONY: bootstrap lint

bootstrap:
	$(UV) sync --frozen

lint:
	$(UV) run ruff check .
