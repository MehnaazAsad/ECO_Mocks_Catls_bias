.PHONY: clean lint test_environment environment update_environment 
	remove_environment catl_mr_make delete_mock_catls delete_all_but_raw
	clean_data_dir test_files delete_catl_screens

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = ECO_Mocks_Catls
PYTHON_INTERPRETER = python3
ENVIRONMENT_FILE = environment.yml
ENVIRONMENT_NAME = eco_mocks_catls

DATA_DIR = $(PROJECT_DIR)/data
SRC_DIR = $(PROJECT_DIR)/src/data
MOCKS_CATL_DIR = $(DATA_DIR)/processed/*

# CPU-Fraction
CPU_FRAC = 0.75
REMOVE_FILES = "False"
COSMO_CHOICE="Planck"
HALOTYPE="m200b"
SURVEY="ECO"
HB_LOCAL="False"
HB_PATH="/fs1/masad/Research/Repositories/ECO_Mocks_Catls/data/raw/hb_files/m200b/ECO"


ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Deletes all build, test, coverage, and Python artifacts
clean: clean-build clean-pyc clean-test

## Removes Python file artifacts
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

## Remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

## Remove test and coverage artifacts
clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache


## Lint using flake8
lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Set up python interpreter environment - Using environment.yml
environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
		# conda config --add channels conda-forge
		conda env create -f $(ENVIRONMENT_FILE)
endif

## Update python interpreter environment
update_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
		conda env update -f $(ENVIRONMENT_FILE)
endif

## Delete python interpreter environment
remove_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, removing conda environment"
		conda env remove -n $(ENVIRONMENT_NAME)
endif

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Create catalogues for `ECO` and/or RESOLVE `A`/`B`
catl_mr_make:
	@python $(SRC_DIR)/mocks_create_main.py -abopt mr -cpu $(CPU_FRAC) -remove $(REMOVE_FILES) -cosmo $(COSMO_CHOICE) -halotype $(HALOTYPE) -survey $(SURVEY) -hb_local $(HB_LOCAL) -hb_path $(HB_PATH)

## Delete existing `mock` catalogues
delete_mock_catls:
	find $(MOCKS_CATL_DIR) -type f -name '*.hdf5' -delete
	find $(MOCKS_CATL_DIR) -type f -name '*.gz' -delete

## Delete all files, except for `raw` files
delete_all_but_raw:
	@rm -rf $(DATA_DIR)/external/*
	@rm -rf $(DATA_DIR)/interim/*
	@rm -rf $(DATA_DIR)/processed/*

## Clean the `./data` folder and remove all of the files
clean_data_dir:
	@rm -rf $(DATA_DIR)/external/*
	@rm -rf $(DATA_DIR)/interim/*
	@rm -rf $(DATA_DIR)/processed/*
	@rm -rf $(DATA_DIR)/raw/*

## Run tests to see if all files (Halobias, catalogues) are in order
test_files:
	@pytest

## Delete screens from creating catalogues
delete_catl_screens:
	screen -S "ECO_RESOLVE_Mocks_create_m200b" -X quit
	screen -S "ECO_RESOLVE_Mocks_create_mvir" -X quit


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
