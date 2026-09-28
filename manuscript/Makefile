PYTHON ?= ../.venv/bin/python

.PHONY: all figures clean
all: main.pdf

figures:
	MPLCONFIGDIR=/tmp/dsa05-manuscript-mpl $(PYTHON) scripts/build_assets.py
	MPLCONFIGDIR=/tmp/dsa05-manuscript-mpl $(PYTHON) scripts/build_diagrams.py

main.pdf: main.tex references.bib $(wildcard sections/*.tex) figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
	cp build/main.pdf main.pdf

clean:
	rm -rf build
