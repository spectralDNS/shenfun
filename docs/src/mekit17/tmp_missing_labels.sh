#rm -rf tmp_mako__* tmp_preprocess__*
find . -name '*.do.txt' -exec grep --color --with-filename --line-number "{fig:poisson2D}" {} \;
