set -eu

# unit test
coverage run --source=mvc -m pytest tests -p no:warnings

# pep8 check
pycodestyle mvc

# additional style check
pylint mvc
