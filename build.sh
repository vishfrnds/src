# To run
# source build.sh

PYTHON_VERSION=3.11.10 # Sets python version
# and installs requirements.txt in a virtual environment activating it


if [ ! -d "$HOME/.pyenv" ]; then
  echo "pyenv not found. Installing pyenv..."
  curl https://pyenv.run | bash
fi
export PATH="$HOME/.pyenv/bin:$PATH"

eval "$($HOME/.pyenv/bin/pyenv init --path)"
eval "$($HOME/.pyenv/bin/pyenv init -)"
eval "$($HOME/.pyenv/bin/pyenv virtualenv-init -)"

PYENV_PATH="$HOME/.pyenv/bin/pyenv"
if ! "$PYENV_PATH" versions --bare | grep -q "^${PYTHON_VERSION}$"; then
    echo "Installing Python $PYTHON_VERSION via pyenv..."
    "$PYENV_PATH" install $PYTHON_VERSION
fi

"$PYENV_PATH" local $PYTHON_VERSION

if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

pip install -r requirements.txt