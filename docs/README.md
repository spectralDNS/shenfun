Some reminders on how to build the documentation
================================================

The documentation is hosted by readthedocs.org

The documentation is built from docstrings and rst-files in the source folder. 
However, the demos (see demos folder) are written using doconce, and corresponding
rst-files under source (poisson.rst, poisson3d.rst, kleingordon.rst, kuramatosivashinsky.rst)
are generated with

    make html

As of now this is not done by readthedocs (because doconce requires a lot of dependencies and 
as such the build takes too long) so remember to run make html locally before pushing
to github. This updates the generated files under source.

If you have made no changes to the demos, then simply

    make ohtml

is faster as it does not regenerate the demos.

To generate a pdf do

    make pdf

However, the pdf looks a bit weird at the time being. 

To run doctests

    make doctest
    
