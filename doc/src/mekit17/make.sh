set -x

name=shenfun

doconce format pdflatex $name "--latex_code_style=default:lst[style=yellow2_fb]@sys:lst-gray"
pdflatex $name
bibtex $name
pdflatex $name

doconce format html $name --html_style=bootswatch_readable --html_code_style=inherit --html_output=shenfun
doconce format html $name --html_style=solarized3 --html_output=shenfun-solarized

doconce format sphinx $name
theme=cloud
#theme=fenics_minimal
doconce sphinx_dir dirname=sphinx-rootdir theme=$theme $name

python automake_sphinx.py

# Publish
dest=../../pub/mekit17
cp -r sphinx-rootdir/_build/html $dest
cp $name.pdf $dest/shenfun.pdf
cp -r figs $dest
cp shenfun*.html $dest

# Add to git if new files have been created
git add $dest

