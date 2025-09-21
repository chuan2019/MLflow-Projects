#!/bin/bash

# Makefile for CartPole DQN Project
# 
# DEVELOPMENT APPROACH:
# - Primary: Docker containers (train, serve, test-api)
# - Optional: Local virtual environment (*-local targets) for IDE support only
# 
# Recommended workflow:
# 1. make setup          # Create venv (optional) + start MLflow
# 2. make train          # Train using Docker
# 3. make serve          # Serve using Docker
# 4. make test-api       # Test the API

.PHONY: help setup train serve clean logs stop status restart build test-all workspace-setup venv-create venv-install test-local

# Virtual environment settings
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_ACTIVATE := source $(VENV_DIR)/bin/activate

help:  ## Show this help message
	@echo "CartPole DQN with MLflow - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv-create:  ## Create virtual environment with uv
	@echo "Creating virtual environment..."
	./venv.sh create

venv-install:  ## Install dependencies in virtual environment
	@echo "Installing dependencies..."
	./venv.sh install

setup: venv-create  ## Setup the project (create venv and start MLflow server)
	@echo "Starting MLflow tracking server..."
	docker-compose up mlflow-server -d
	@echo "MLflow UI available at http://localhost:5000"

train:  ## Train the DQN model using Docker
	@echo "Starting DQN training..."
	docker-compose --profile training up cartpole-trainer

train-local:  ## Train locally using virtual environment
	@echo "Starting local DQN training..."
	./venv.sh run python src/train.py

serve:  ## Serve the trained model using Docker
	@echo "Starting model server..."
	docker-compose --profile serving up model-server

serve-local:  ## Serve model locally using virtual environment
	@echo "Starting local model server..."
	./venv.sh run python src/serve.py

test-api:  ## Test the model API
	@echo "Testing model API..."
	@echo "Health check:"
	curl -s http://localhost:8080/health | python -m json.tool
	@echo "\nModel info:"
	curl -s http://localhost:8080/model_info | python -m json.tool
	@echo "\nPrediction test:"
	curl -s -X POST http://localhost:8080/predict \
		-H "Content-Type: application/json" \
		-d '{"state": [0.1, 0.2, 0.3, 0.4]}' | python -m json.tool

test-local:  ## Run comprehensive test suite
	@echo "Running test suite..."
	./venv.sh test

test-unit:  ## Run unit tests with pytest
	@echo "Running unit tests..."
	./venv.sh run pytest

format:  ## Format code with black
	@echo "Formatting code..."
	./venv.sh run black src/ test_suite.py

lint:  ## Lint code with flake8
	@echo "Linting code..."
	./venv.sh run flake8 src/

logs:  ## Show logs for all services
	docker-compose logs -f

stop:  ## Stop all services
	@echo "Stopping all services..."
	docker-compose down
	@echo "All services stopped"

status:  ## Show status of all services
	@echo "Service status:"
	docker-compose ps

restart:  ## Restart all services
	@echo "Restarting services..."
	docker-compose restart

build:  ## Build all Docker images
	@echo "Building Docker images..."
	docker-compose build

workspace-setup:  ## Complete workspace setup (build + start MLflow)
	@echo "Setting up complete workspace..."
	@make build
	@make setup
	@echo "Workspace ready! MLflow UI: http://localhost:5000"

down:  ## Stop all services and remove containers
	@echo "Stopping all services..."
	docker-compose down
	@echo "All services stopped"

clean:  ## Clean up Docker containers, volumes, and virtual environment
	@echo "Stopping all services (including profiles)..."
	docker-compose --profile training --profile serving down
	@echo "Removing volumes and networks..."
	docker-compose down -v --remove-orphans
	@echo "Pruning Docker system..."
	docker system prune -f
	@echo "Docker cleanup complete"
	@echo "Cleaning up virtual environment..."
	./venv.sh clean
	@echo "Virtual environment cleanup complete"

install:  ## Install Python dependencies in virtual environment
	@echo "Installing dependencies..."
	./venv.sh install

notebook:  ## Start Jupyter Lab
	@echo "Starting Jupyter Lab..."
	./venv.sh run jupyter lab

mlflow-ui:  ## Open MLflow UI in browser
	@echo "Opening MLflow UI..."
	open http://localhost:5000 || xdg-open http://localhost:5000 || echo "Please open http://localhost:5000 in your browser"

# Development targets
dev-setup: venv-create setup  ## Setup for local development
	@echo "Development environment ready!"

full-run: setup train serve  ## Complete pipeline: setup -> train -> serve
	@echo "Full pipeline completed!"

# Quick commands
quick-train:  ## Quick training with fewer episodes
	./venv.sh run python src/train.py --episodes 100 --batch-size 16

demo:  ## Run a quick demo
	@echo "Running CartPole DQN demo..."
	@make setup
	@sleep 5
	@make quick-train
	@make serve-local &
	@sleep 10
	@make test-api

test-all:  ## Run comprehensive tests (build, train, serve, test)
	@bash -c ' \
	set -e; \
	trap "make down" EXIT; \
	echo "Running comprehensive test suite..."; \
	make workspace-setup; \
	echo "Starting training..."; \
	make train; \
	echo "Starting model server..."; \
	make serve & \
	SERVER_PID=$$!; \
	sleep 15; \
	echo "Testing API..."; \
	make test-api; \
	echo "All tests completed successfully!"; \
	'
