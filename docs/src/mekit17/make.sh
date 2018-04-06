set -x

name=shenfun

doconce format pdflatex $name "--latex_code_style=default:lst[style=yellow2_fb]@sys:lst-gray"
pdflatex $name
bibtex $name
pdflatex $name

doconce format html $name --html_style=solarized3 --html_output=shenfun-solarized

style=bootstrap
doconce format html $name --html_style=${style} --pygments_html_style=default --html_admon=bootstrap_alert --html_output=shenfun_${style} --keep_pygments_html_bg —html_code_style=inherit --html_pre_style=inherit
doconce split_html shenfun_${style}.html

style=bootswatch_cyborg
doconce format html $name --html_style=${style} --pygments_html_style=monokai --html_admon=bootstrap_alert --html_output=${name}_${style} --keep_pygments_html_bg --html_code_style=inherit --html_pre_style=inherit
doconce split_html ${name}_${style}.html

doconce format sphinx $name
theme=bloodish
#theme=uio2
doconce sphinx_dir dirname=sphinx-rootdir theme=$theme $name

python automake_sphinx.py

# Publish
dest=pub
cp -r sphinx-rootdir/_build/html/ $dest
cp $name.pdf $dest/shenfun.pdf
cp -r figs $dest
cp shenfun*.html $dest/
cp ._shenfun* $dest

# Add to git if new files have been created
git add $dest
git reset -- $dest/src

