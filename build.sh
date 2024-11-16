# To run
# source build.sh

if [ "$VIRTUAL_ENV" != "" ]; then
    echo "Deactivating virtual environment"
    deactivate
fi

if [ "$1" = "--install" ] || [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r src/requirements.txt
else
    source venv/bin/activate
fi

export PYTHONPATH=.