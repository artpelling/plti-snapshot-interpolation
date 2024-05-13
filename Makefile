#!/usr/bin/env make

.PHONY: clean plots

clean :
	-rm ./figures/*.pdf
	-rm -rf ./venv

venv : requirements.txt
	virtualenv venv -p=3.10
	./venv/bin/python -m pip install --upgrade pip
	./venv/bin/python -m pip install -r requirements.txt
	@touch venv

figures : venv
	./venv/bin/python example_toy.py
	./venv/bin/python example_polynomial.py
	./venv/bin/python example_penzl.py
