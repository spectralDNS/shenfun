VERSION=$(shell python -c "import shenfun; print(shenfun.__version__)")

default:
	python setup.py build_ext -i

tag:
	git tag $(VERSION)
	git push --tags

clean:
	git clean shenfun -fx
	git clean tests -fx
	cd docs && make clean && cd ..
	@rm -rf *.egg-info/ build/ dist/ .eggs/