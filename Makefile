SHELL := /bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

.PHONY: help lint local build pull run

.DEFAULT: help

help:
	@echo "make run"
	@echo "	run python main.py"

ENV_NAME := super-res
run:
	@cd src;\
	$(CONDA_ACTIVATE) $(ENV_NAME);\
	python main.py;\
	cd ..;