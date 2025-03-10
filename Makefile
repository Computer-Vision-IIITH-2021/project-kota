SHELL := /bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

.PHONY: help lint local build pull run

.DEFAULT: help

help:
	@echo "make run"
	@echo "	run python main.py"

ENV_NAME := super-res
visdom:
	@cd src;\
	$(CONDA_ACTIVATE) $(ENV_NAME);\
	python -m visdom.server;\

run:
	@cd src;\
	$(CONDA_ACTIVATE) $(ENV_NAME);\
 	CUDA_VISIBLE_DEVICES=2,3 nohup python main.py > main.log &
	cd ..;
