# Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
PORT          ?= 8000

.PHONY: help html serve clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "Documentation build complete. Output in $(BUILDDIR)/html"

serve: html
	@echo "Starting documentation preview server on http://localhost:$(PORT)"
	@python3 -m http.server $(PORT) --directory "$(BUILDDIR)/html"

clean:
	rm -rf "$(BUILDDIR)"
