.PHONY:	help build up

.DEFAULT_GOAL := help

run_unit_tests: ## Run the unit tests
				python3.7 -m unittest discover tests/unit_tests

build_whl_file: ## Build the whl file
				python3.7 setup.py sdist bdist_wheel