# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line. 
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = shenfun
SOURCEDIR     = source
BUILDDIR      = build
DEMO          = Poisson/poisson.rst \
                KleinGordon/kleingordon.rst \
                Poisson3D/poisson3d.rst \
		KuramatoSivashinsky/kuramatosivashinsky.rst

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%.rst:
	doconce format sphinx src/$(basename $@).do.txt
	cp src/$(basename $@).rst source/
	./recite.sh $(notdir $@)
	if [ -d src/$(dir $@)figs ]; then cp src/$(dir $@)figs/*.* source/figs/; fi
	if [ -d src/$(dir $@)movies ]; then cp src/$(dir $@)movies/*.* source/movies/; fi

publish:	
	cd src && publish export papers.bib && cd ..
	cp src/papers.bib source/

demos: $(DEMO)

pdf:
	@$(SPHINXBUILD) -b latex -d "$(BUILDDIR)"/doctrees $(SOURCEDIR) "$(BUILDDIR)"/latex
	$(MAKE) -C "$(BUILDDIR)"/latex all-pdf LATEXMKOPTS='-interaction=nonstopmode -f'

html: publish $(DEMO)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

