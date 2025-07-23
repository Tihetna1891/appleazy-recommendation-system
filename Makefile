# Makefile to clean cache recursively

.PHONY: clean

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 
	find . -type f -name "*.pyc" -exec rm -f {} +