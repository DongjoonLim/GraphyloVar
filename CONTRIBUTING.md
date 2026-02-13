# Contributing to GraphyloVar

Thank you for your interest in contributing to GraphyloVar! This document provides guidelines and best practices for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GraphyloVar.git
   cd GraphyloVar
   ```
3. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Using Conda (Recommended)

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml
conda activate graphylovar

# Install development dependencies
pip install pytest flake8 black mypy
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest flake8 black mypy
```

## Code Style

We follow PEP 8 style guidelines with some modifications:

- **Line length**: Maximum 120 characters
- **Formatting**: Use `black` for automatic code formatting
- **Imports**: Use `isort` to organize imports
- **Type hints**: Add type hints to all function signatures
- **Docstrings**: Use Google-style or NumPy-style docstrings

### Running Code Formatters

```bash
# Format code with black
black .

# Check code style with flake8
flake8 .

# Check type hints with mypy
mypy *.py
```

## Testing

We use `pytest` for testing. All new features should include tests.

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_utils.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Writing Tests

- Place test files in the `tests/` directory
- Name test files with `test_` prefix (e.g., `test_utils.py`)
- Use descriptive test names that explain what is being tested
- Include docstrings in test methods
- Test both normal and edge cases

Example:

```python
def test_reverse_complement_simple_sequence():
    """Test reverse complement of simple DNA sequence."""
    sequence = ["A", "T", "C", "G"]
    expected = ["C", "G", "A", "T"]
    assert reverse_complement(sequence) == expected
```

## Submitting Changes

1. **Ensure tests pass**: Run the full test suite before submitting
2. **Format code**: Run `black` and `flake8` on your changes
3. **Commit changes**: Use clear, descriptive commit messages
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```
4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Create Pull Request**: Go to the original repository and create a PR from your fork

### Pull Request Guidelines

- **Title**: Clear and descriptive (e.g., "Add unit tests for preprocessing functions")
- **Description**: Explain what changes were made and why
- **Reference issues**: Link to related issues (e.g., "Fixes #123")
- **Keep it focused**: One feature or fix per PR
- **Update documentation**: If your changes affect usage, update README.md

## Reporting Bugs

When reporting bugs, please include:

1. **Clear title**: Brief description of the bug
2. **Environment**: Python version, OS, package versions
3. **Steps to reproduce**: Detailed steps to reproduce the bug
4. **Expected behavior**: What you expected to happen
5. **Actual behavior**: What actually happened
6. **Error messages**: Full error messages and stack traces
7. **Code samples**: Minimal code to reproduce the issue

Use the GitHub issue template if available.

## Feature Requests

We welcome feature requests! Please include:

1. **Clear description**: What feature you'd like to see
2. **Use case**: Why this feature would be useful
3. **Proposed solution**: (Optional) How you think it could be implemented
4. **Alternatives**: (Optional) Alternative solutions you've considered

## Development Guidelines

### Adding New Preprocessing Scripts

- Place scripts in the root directory
- Use the shared `config.py` for constants
- Use utility functions from `utils.py` for common operations
- Add command-line argument parsing with `argparse`
- Include comprehensive docstrings
- Add logging using the `logging` module

### Adding New Models

- Place model architectures in training scripts or create a `models.py` module
- Document architecture in docstrings
- Use configuration constants from `config.py`
- Include model summaries in training output
- Save model checkpoints and training history

### Documentation

- Update README.md for user-facing changes
- Add docstrings to all functions and classes
- Include examples in docstrings where helpful
- Update this CONTRIBUTING.md if workflow changes

## Questions?

If you have questions about contributing, please:

1. Check existing documentation
2. Search existing issues on GitHub
3. Open a new issue with your question

Thank you for contributing to GraphyloVar!
