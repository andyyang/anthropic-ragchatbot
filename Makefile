# Development commands for code quality

.PHONY: format lint test quality install clean help

# Format code with Black and isort
format:
	@echo "🎨 Formatting Python code..."
	uv run autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive backend/
	uv run black .
	uv run isort .
	@echo "✅ Code formatting complete!"

# Run linting checks
lint:
	@echo "🔍 Running code quality checks..."
	uv run flake8 backend/ --max-line-length=88 --extend-ignore=E203,W503,E501,E402,F811,W291,F841
	uv run black --check .
	uv run isort --check-only .
	@echo "✅ Linting complete!"

# Run tests with coverage
test:
	@echo "🧪 Running tests with coverage..."
	uv run pytest
	@echo "✅ Tests complete!"

# Run all quality checks
quality: format lint test
	@echo "🎉 All quality checks complete!"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	uv sync
	@echo "✅ Dependencies installed!"

# Clean up temporary files
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	@echo "✅ Cleanup complete!"

# Show available commands
help:
	@echo "📋 Available commands:"
	@echo "  make format   - Format code with Black and isort"
	@echo "  make lint     - Run linting checks"
	@echo "  make test     - Run tests with coverage"
	@echo "  make quality  - Run all quality checks"
	@echo "  make install  - Install dependencies"
	@echo "  make clean    - Clean up temporary files"
	@echo "  make help     - Show this help message"