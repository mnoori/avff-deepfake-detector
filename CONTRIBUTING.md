# Contributing to AVFF Deepfake Detector

Thank you for your interest in contributing to the AVFF Deepfake Detector project! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others.

## How to Contribute

1. **Fork the Repository**
   - Click the 'Fork' button on the GitHub repository page
   - Clone your forked repository to your local machine

2. **Create a New Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow the project's coding style
   - Add tests for new features
   - Update documentation as needed

4. **Commit Your Changes**
   ```bash
   git commit -m "Description of your changes"
   ```

5. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Go to the original repository
   - Click 'New Pull Request'
   - Select your branch
   - Add a description of your changes
   - Submit the pull request

## Development Setup

1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Coding Standards

- Follow PEP 8 style guide
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting a pull request
- Run tests locally:
  ```bash
  pytest
  ```

## Documentation

- Update README.md for significant changes
- Add docstrings to new functions and classes
- Keep comments clear and concise

## Pull Request Process

1. Ensure your code passes all tests
2. Update documentation as needed
3. Describe your changes in the pull request
4. Wait for review and address any feedback

## Questions?

Feel free to open an issue if you have any questions about contributing to the project. 