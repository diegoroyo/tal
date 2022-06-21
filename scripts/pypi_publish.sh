#!/bin/bash

echo "You should probably execute test_pypi_publish.sh first. OK? (y):"
read a
if [[ $a != 'y' ]]
then
    exit
fi

twine upload dist/*