RM = rm -rf

# all: docker-build

sdist:
	@echo "Building python project..."
	@python setup.py sdist

pip-install-edit-mode:
	@pip install -e .

re: fclean all

fclean:
	# https://stackoverflow.com/questions/10722723/find-exec-echo-missing-argument-to-exec
	@find . -name "*.pyc" -exec git rm --cached {} \;
	@find . -type d -name "__pycache__" -exec rm -Rf {} \;
	@rm -r $(find . -name '*.egg-info')