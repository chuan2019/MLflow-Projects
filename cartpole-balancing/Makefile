#!/bin/bash

# Makefile for CartPole DQN Project

.PHONY: help setup train serve clean logs stop

help:  ## Show this help message
	@echo "CartPole DQN with MLflow - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Setup the project (start MLflow server)
	@echo "Starting MLflow tracking server..."
	docker-compose up mlflow-server -d
	@echo "MLflow UI available at http://localhost:5000"

train:  ## Train the DQN model
	@echo "Starting DQN training..."
	docker-compose --profile training up cartpole-trainer

train-local:  ## Train locally (requires Python environment)
	@echo "Starting local DQN training..."
	python src/train.py

serve:  ## Serve the trained model
	@echo "Starting model server..."
	docker-compose --profile serving up model-server

serve-local:  ## Serve model locally
	@echo "Starting local model server..."
	python src/serve.py

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

logs:  ## Show logs for all services
	docker-compose logs -f

stop:  ## Stop all services
	@echo "Stopping all services..."
	docker-compose down
	@echo "All services stopped"

clean:  ## Clean up Docker containers and volumes
	@echo "Cleaning up..."
	docker-compose down -v
	docker system prune -f
	@echo "Cleanup complete"

install:  ## Install Python dependencies locally
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed"

notebook:  ## Start Jupyter notebook
	@echo "Starting Jupyter notebook..."
	jupyter notebook notebooks/

mlflow-ui:  ## Open MLflow UI in browser
	@echo "Opening MLflow UI..."
	open http://localhost:5000 || xdg-open http://localhost:5000 || echo "Please open http://localhost:5000 in your browser"

# Development targets
dev-setup: install setup  ## Setup for local development
	@echo "Development environment ready!"

full-run: setup train serve  ## Complete pipeline: setup -> train -> serve
	@echo "Full pipeline completed!"

# Quick commands
quick-train:  ## Quick training with fewer episodes
	python src/train.py --episodes 100 --batch-size 16

demo:  ## Run a quick demo
	@echo "Running CartPole DQN demo..."
	@make setup
	@sleep 5
	@make quick-train
	@make serve-local &
	@sleep 10
	@make test-api