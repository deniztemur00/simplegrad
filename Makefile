VENV=.venv
BIN=$(VENV)/bin
PY_FOLDER =py-simplegrad

.venv:
	python3 -m venv $(VENV)
	$(MAKE) requirements

.PHONY: requirements
requirements: .venv
	@unset CONDA_PREFIX \
 && $(BIN)/python -m pip install --upgrade pip \
 &&	$(BIN)/python -m pip install -r requirements.txt
##  && $(BIN)/activate

.PHONY: install
install:
	$(BIN)/python -m pip install .

.PHONY: build
build: .venv
	mkdir -p build && cd build && cmake .. && make
	$(MAKE) install

.PHONY: build-python
build-python: .venv
	$(BIN)/python -m pip uninstall -y simplegrad
	$(MAKE) clean
	$(BIN)/python setup.py sdist bdist_wheel --plat-name "manylinux2014_x86_64"
	$(BIN)/python -m pip install .



.PHONY: build-release
build-release: .venv
	mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make
	$(MAKE) install
	
.PHONY: setup
setup: .venv
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf buid/
	$(BIN)/python setup.py install

.PHONY: test
test:
	$(BIN)/pytest py-simplegrad/tests -v

.PHONY: run
run:
	$(BIN)/python $(PY_FOLDER)/main.py

.PHONY: publish-test
publish-test:
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf buid/
	$(BIN)/python setup.py clean --all
	$(BIN)/python setup.py sdist bdist_wheel --plat-name "manylinux2014_x86_64"
	$(BIN)/python -m twine upload --repository testpypi dist/* --verbose --skip-existing

.PHONY: publish
publish:
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf buid/
	$(BIN)/python setup.py clean --all
	$(BIN)/python setup.py sdist bdist_wheel --plat-name "manylinux2014_x86_64"
	$(BIN)/python -m twine upload dist/* --verbose --skip-existing

.PHONY: clean
clean:
#	rm -rf $(VENV)
	rm -rf build
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .hypothesis/

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make set            - Set up the virtual environment and install requirements"
	@echo "  make install        - Install the package in editable mode"
	@echo "  make build          - Build the project"
	@echo "  make build-release  - Build the project in release mode"
	@echo "  make build-python   - Build the project in release mode and install it"
	@echo "  make test           - Run tests using pytest"
	@echo "  make run            - Run the main Python script"
	@echo "  make clean          - Clean up the build and virtual environment"
	@echo "  make help           - Show this help message"


