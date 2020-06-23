VERSION=$(shell python3 -c "import shenfun; print(shenfun.__version__)")

default:
	python setup.py build_ext -i

pip:
	rm -f dist/*
	python setup.py sdist
	twine upload dist/*

tag:
	git tag -f $(VERSION)
	git push --tags

publish: tag pip

clean:
	git clean shenfun -fx
	git clean tests -fx
	cd docs && make clean && cd ..
	@rm -rf *.egg-info/ build/ dist/ .eggs/