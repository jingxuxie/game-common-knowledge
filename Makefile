.PHONY: test experiments validate paper aaai clean

PYTHON ?= python
PDFLATEX ?= pdflatex
BIBTEX ?= $(shell if command -v bibtex >/dev/null 2>&1; then echo bibtex; else echo bibtex8; fi)
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

# Source imports also work without editable installation.
test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

experiments:
	PYTHONPATH=src $(PYTHON) experiments/run_all.py

validate: experiments
	PYTHONPATH=src $(PYTHON) scripts/validate_results.py

paper: validate
	cd paper && $(PDFLATEX) $(LATEXFLAGS) preprint.tex
	cd paper && $(BIBTEX) preprint
	cd paper && $(PDFLATEX) $(LATEXFLAGS) preprint.tex
	cd paper && $(PDFLATEX) $(LATEXFLAGS) preprint.tex
	cd paper && $(PDFLATEX) $(LATEXFLAGS) supplement.tex
	cd paper && $(BIBTEX) supplement
	cd paper && $(PDFLATEX) $(LATEXFLAGS) supplement.tex
	cd paper && $(PDFLATEX) $(LATEXFLAGS) supplement.tex

aaai: validate
	@test -f paper/aaai2027.sty || (echo "Download the official AAAI-27 Author Kit and place aaai2027.sty in paper/."; exit 1)
	@test -f paper/aaai2027.bst || (echo "Download the official AAAI-27 Author Kit and place aaai2027.bst in paper/."; exit 1)
	cd paper && $(PDFLATEX) $(LATEXFLAGS) main.tex
	cd paper && $(BIBTEX) main
	cd paper && $(PDFLATEX) $(LATEXFLAGS) main.tex
	cd paper && $(PDFLATEX) $(LATEXFLAGS) main.tex

clean:
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.fdb_latexmk paper/*.fls \
		paper/*.log paper/*.out paper/*.toc paper/*.pdf
