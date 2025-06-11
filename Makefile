# === CONFIG ===
VENV_DIR = .venv
PYTHON = python3.10

# === COLORS ===
BLUE := \033[36m
BOLD := \033[1m
RESET := \033[0m

.DEFAULT_GOAL := help
.PHONY: help venv install train test format clean

##@ Setup

venv:  ## Create a virtual environment
	@printf "$(BLUE)Creating virtual environment...$(RESET)\n"
	$(PYTHON) -m venv $(VENV_DIR)

install: venv  ## Install dependencies using pip
	@printf "$(BLUE)Installing dependencies...$(RESET)\n"
	. $(VENV_DIR)/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt && \
	pip install -e .

##@ Execution

train: install  ## Train the RL agent
	. $(VENV_DIR)/bin/activate && python scripts/run_train.py

test: install  ## Test the RL agent
	. $(VENV_DIR)/bin/activate && python scripts/run_test.py

format:  ## Format code using black
	. $(VENV_DIR)/bin/activate && pip install black && \
	black src/ tests/ scripts/

##@ Maintenance

clean:  ## Remove the virtual environment
	@printf "$(BLUE)Cleaning project...$(RESET)\n"
	rm -rf $(VENV_DIR)

##@ Help

help:  ## Display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
