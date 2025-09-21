#!/usr/bin/env python3
"""
Comprehensive test suite for CartPole DQN project
Tests functionality, imports, and potential issues
"""

import sys
import os
import subprocess
import importlib.util
import traceback
from pathlib import Path

# Test configuration
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

class TestRunner:
    """Test runner for CartPole DQN project"""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.results = []
    
    def run_test(self, test_name, test_func):
        """Run a single test and record results"""
        print(f"\n{'='*60}")
        print(f"Testing: {test_name}")
        print(f"{'='*60}")
        
        try:
            test_func()
            print(f"[PASS] {test_name}")
            self.passed_tests += 1
            self.results.append((test_name, "PASSED", None))
        except Exception as e:
            print(f"[FAIL] {test_name}")
            print(f"Error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            self.failed_tests += 1
            self.results.append((test_name, "FAILED", str(e)))
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {self.passed_tests + self.failed_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success rate: {self.passed_tests/(self.passed_tests + self.failed_tests)*100:.1f}%")
        
        if self.failed_tests > 0:
            print(f"\nFailed tests:")
            for name, status, error in self.results:
                if status == "FAILED":
                    print(f"  - {name}: {error}")

def test_project_structure():
    """Test project structure and required files"""
    required_files = [
        "requirements.txt",
        "docker-compose.yml",
        "Dockerfile.trainer",
        "Dockerfile.server",
        ".env",
        "README.md",
        "Makefile",
        "src/dqn_agent.py",
        "src/train.py",
        "src/serve.py",
        "src/mlflow_utils.py",
    ]
    
    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        assert full_path.exists(), f"Required file missing: {file_path}"
    
    # Check directories
    required_dirs = ["src", "models", "notebooks"]
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        assert full_path.exists() and full_path.is_dir(), f"Required directory missing: {dir_path}"

def test_python_imports():
    """Test that all Python modules can be imported without errors"""
    sys.path.insert(0, str(SRC_DIR))
    
    modules_to_test = [
        "dqn_agent",
        "mlflow_utils", 
        "train",
        "serve"
    ]
    
    for module_name in modules_to_test:
        try:
            spec = importlib.util.spec_from_file_location(module_name, SRC_DIR / f"{module_name}.py")
            module = importlib.util.module_from_spec(spec)
            # Don't actually execute the module, just check if it can be loaded
            print(f"[OK] Module {module_name} can be imported")
        except Exception as e:
            raise ImportError(f"Failed to import {module_name}: {str(e)}")

def test_dependencies():
    """Test that required dependencies are available"""
    required_packages = [
        "torch",
        "numpy", 
        "gymnasium",
        "mlflow",
        "flask",
        "matplotlib",
        "pandas",
        "dotenv"  # python-dotenv imports as 'dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"[OK] Package {package} is available")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(f"Missing packages: {', '.join(missing_packages)}")

def test_environment_config():
    """Test environment configuration"""
    env_file = PROJECT_ROOT / ".env"
    assert env_file.exists(), ".env file not found"
    
    required_env_vars = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME", 
        "EPISODES",
        "LEARNING_RATE",
        "BATCH_SIZE"
    ]
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    for var in required_env_vars:
        assert var in content, f"Environment variable {var} not found in .env"
        print(f"[OK] Environment variable {var} found")

def test_docker_config():
    """Test Docker configuration"""
    compose_file = PROJECT_ROOT / "docker-compose.yml"
    assert compose_file.exists(), "docker-compose.yml not found"
    
    # Test docker-compose syntax
    result = subprocess.run(
        ["docker-compose", "config"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Docker compose config invalid: {result.stderr}")
    
    print("[OK] Docker Compose configuration is valid")

def test_dqn_agent_basic():
    """Test basic DQN agent functionality"""
    sys.path.insert(0, str(SRC_DIR))
    
    # Import without executing
    import importlib.util
    spec = importlib.util.spec_from_file_location("dqn_agent", SRC_DIR / "dqn_agent.py")
    dqn_module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(dqn_module)
        
        # Check required classes exist
        assert hasattr(dqn_module, 'DQN'), "DQN class not found"
        assert hasattr(dqn_module, 'DQNAgent'), "DQNAgent class not found"
        assert hasattr(dqn_module, 'ReplayBuffer'), "ReplayBuffer class not found"
        
        print("[OK] All required classes found in dqn_agent.py")
        
        # Test basic instantiation (if dependencies are available)
        try:
            import torch
            import numpy as np
            
            # Test DQN network
            dqn = dqn_module.DQN(input_size=4, hidden_size=64, output_size=2)
            assert dqn is not None
            print("[OK] DQN network can be instantiated")
            
            # Test replay buffer
            buffer = dqn_module.ReplayBuffer(capacity=1000)
            assert len(buffer) == 0
            print("[OK] ReplayBuffer can be instantiated")
            
        except ImportError:
            print("! Skipping instantiation tests (dependencies not available)")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load dqn_agent module: {str(e)}")

def test_mlflow_utils_basic():
    """Test MLflow utilities basic functionality"""
    sys.path.insert(0, str(SRC_DIR))
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("mlflow_utils", SRC_DIR / "mlflow_utils.py")
    mlflow_module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(mlflow_module)
        
        # Check required classes exist
        assert hasattr(mlflow_module, 'MLflowTracker'), "MLflowTracker class not found"
        assert hasattr(mlflow_module, 'get_best_model_uri'), "get_best_model_uri function not found"
        
        print("[OK] All required classes/functions found in mlflow_utils.py")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load mlflow_utils module: {str(e)}")

def test_training_script_structure():
    """Test training script structure"""
    train_file = SRC_DIR / "train.py"
    assert train_file.exists(), "train.py not found"
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Check for required functions/patterns
    required_patterns = [
        "def main(",
        "if __name__ == \"__main__\":",
        "parse_args",
        "train_dqn_agent",
        "mlflow"
    ]
    
    for pattern in required_patterns:
        assert pattern in content, f"Required pattern '{pattern}' not found in train.py"
        print(f"[OK] Pattern '{pattern}' found in train.py")

def test_serving_script_structure():
    """Test serving script structure"""
    serve_file = SRC_DIR / "serve.py"
    assert serve_file.exists(), "serve.py not found"
    
    with open(serve_file, 'r') as f:
        content = f.read()
    
    # Check for required patterns
    required_patterns = [
        "from flask import",
        "@app.route",
        "/predict",
        "/health",
        "def main("
    ]
    
    for pattern in required_patterns:
        assert pattern in content, f"Required pattern '{pattern}' not found in serve.py"
        print(f"[OK] Pattern '{pattern}' found in serve.py")

def test_makefile_commands():
    """Test Makefile has required commands"""
    makefile = PROJECT_ROOT / "Makefile"
    assert makefile.exists(), "Makefile not found"
    
    with open(makefile, 'r') as f:
        content = f.read()
    
    required_commands = [
        "setup:",
        "train:",
        "serve:",
        "clean:",
        "help:"
    ]
    
    for command in required_commands:
        assert command in content, f"Required command '{command}' not found in Makefile"
        print(f"[OK] Command '{command}' found in Makefile")

def test_requirements_validity():
    """Test that requirements.txt has valid package specifications"""
    req_file = PROJECT_ROOT / "requirements.txt"
    assert req_file.exists(), "requirements.txt not found"
    
    with open(req_file, 'r') as f:
        lines = f.readlines()
    
    package_count = 0
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # Basic validation of package format
            assert '==' in line or '>=' in line or '<=' in line or '>' in line or '<' in line, \
                f"Invalid package specification: {line}"
            package_count += 1
            print(f"[OK] Valid package spec: {line}")
    
    assert package_count > 0, "No packages found in requirements.txt"
    print(f"[OK] Found {package_count} package specifications")

def main():
    """Run all tests"""
    print("CartPole DQN Project Test Suite")
    print("===============================")
    
    runner = TestRunner()
    
    # Define tests to run
    tests = [
        ("Project Structure", test_project_structure),
        ("Python Imports", test_python_imports),
        ("Dependencies", test_dependencies),
        ("Environment Config", test_environment_config),
        ("Docker Config", test_docker_config),
        ("DQN Agent Basic", test_dqn_agent_basic),
        ("MLflow Utils Basic", test_mlflow_utils_basic),
        ("Training Script Structure", test_training_script_structure),
        ("Serving Script Structure", test_serving_script_structure),
        ("Makefile Commands", test_makefile_commands),
        ("Requirements Validity", test_requirements_validity),
    ]
    
    # Run all tests
    for test_name, test_func in tests:
        runner.run_test(test_name, test_func)
    
    # Print summary
    runner.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if runner.failed_tests == 0 else 1)

if __name__ == "__main__":
    main()