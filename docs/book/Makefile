.PHONY: help book clean serve

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  book        to build the book"
	@echo "  clean       to clean out site build files"
	@echo "  commit      to build the book and commit to gh-pages online"
	@echo "  pdf         to build the sites PDF"


book:
	jupyter-book build ./

commit:
	jupyter-book build ./
	ghp-import -n -p -f _build/html

clean:
	jupyter-book clean ./

pdf:
	jupyter-book build ./ --builder pdflatex
