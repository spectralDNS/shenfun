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
				PolarHelmholtz/polarhelmholtz.rst \
				KuramatoSivashinsky/kuramatosivashinsky.rst \
				Stokes/stokes.rst DrivenCavity/drivencavity.rst \
				RayleighBenard/rayleighbenard.rst \
				Stokes/stokes.ipynb DrivenCavity/drivencavity.ipynb

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%.rst:
	doconce format sphinx demos/$(basename $@).do.txt --sphinx_preserve_bib_keys
	doconce subst 'XXX' '   ' demos/$(basename $@).rst
	cp demos/$(basename $@).rst source/
	./recite.sh $(notdir $@)

%.ipynb:
	doconce format ipynb demos/$(basename $@).do.txt
	cp demos/$(basename $@).ipynb ../binder/

publish:
	cd demos && publish export papers.bib && cd ..
	cp demos/papers.bib source/

demos: $(DEMO)

pdf:
	@$(SPHINXBUILD) -b latex -d "$(BUILDDIR)"/doctrees $(SOURCEDIR) "$(BUILDDIR)"/latex
	$(MAKE) -C "$(BUILDDIR)"/latex all-pdf LATEXMKOPTS='-interaction=nonstopmode -f'

html: publish $(DEMO)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

ohtml:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

doctest:
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
