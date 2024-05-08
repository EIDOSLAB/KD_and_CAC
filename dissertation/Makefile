TEX=$(patsubst %.tex,%,$(wildcard *.tex))
AUX=$(patsubst %.aux,%,$(wildcard build/*.aux))

all: pdf

pdf: $(TEX).pdf

%.pdf: %.tex
	@mkdir -p build
	@pdflatex -output-directory build $<
	@mv build/*.pdf .

bibliography:
	@biber build/*.bcf

.PHONY: clean $(TEX).tex

clean:
	@rm build/* 2>/dev/null || true
