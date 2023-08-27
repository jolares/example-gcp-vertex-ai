# Installs Python version specified in .python-version file.
pyenv install

# Sets Python version for the active shell.
pyenv shell

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
. .venv/bin/activate

# Installs Python dependencies
pip -r requirements-dev.txt