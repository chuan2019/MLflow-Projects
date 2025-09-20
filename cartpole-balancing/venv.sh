#!/bin/bash

# Virtual Environment Management Script for CartPole DQN Project
# This script provides easy commands to manage the uv virtual environment

set -e

VENV_DIR=".venv"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install it first:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_info "Creating virtual environment with uv..."
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists. Use 'recreate' to recreate it."
        return 1
    fi
    
    uv venv "$VENV_DIR"
    print_success "Virtual environment created at $VENV_DIR"
    
    # Install dependencies
    install_deps
}

# Recreate virtual environment
recreate_venv() {
    print_info "Recreating virtual environment..."
    if [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
        print_info "Removed existing virtual environment"
    fi
    create_venv
}

# Install dependencies
install_deps() {
    print_info "Installing dependencies..."
    if [ ! -d "$VENV_DIR" ]; then
        print_error "Virtual environment not found. Run 'create' first."
        exit 1
    fi
    
    source "$VENV_DIR/bin/activate"
    
    # Install main dependencies
    uv pip install -r requirements.txt
    
    # Install development dependencies
    uv pip install pytest pytest-cov black flake8 mypy jupyterlab
    
    print_success "Dependencies installed successfully"
}

# Activate environment (for manual use)
activate_env() {
    if [ ! -d "$VENV_DIR" ]; then
        print_error "Virtual environment not found. Run 'create' first."
        exit 1
    fi
    
    print_info "To activate the virtual environment, run:"
    echo "  source $VENV_DIR/bin/activate"
}

# Run commands in the virtual environment
run_in_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        print_error "Virtual environment not found. Run 'create' first."
        exit 1
    fi
    
    source "$VENV_DIR/bin/activate"
    "$@"
}

# Test the environment
test_env() {
    print_info "Testing virtual environment..."
    run_in_venv python test_suite.py
}

# Clean up
clean() {
    print_info "Cleaning up..."
    if [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
        print_success "Virtual environment removed"
    fi
    
    # Clean Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Show help
show_help() {
    echo "Virtual Environment Management for CartPole DQN Project"
    echo ""
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  create     Create virtual environment and install dependencies"
    echo "  recreate   Recreate virtual environment from scratch"
    echo "  install    Install/update dependencies in existing environment"
    echo "  activate   Show activation command"
    echo "  run <cmd>  Run command in virtual environment"
    echo "  test       Run test suite in virtual environment"
    echo "  clean      Remove virtual environment and clean cache"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 create                    # Create environment and install deps"
    echo "  $0 run python src/train.py   # Run training in virtual environment"
    echo "  $0 run jupyter lab           # Start Jupyter Lab"
    echo "  $0 test                      # Run tests"
}

# Main script logic
main() {
    cd "$PROJECT_DIR"
    check_uv
    
    case "${1:-help}" in
        create)
            create_venv
            ;;
        recreate)
            recreate_venv
            ;;
        install)
            install_deps
            ;;
        activate)
            activate_env
            ;;
        run)
            shift
            run_in_venv "$@"
            ;;
        test)
            test_env
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"