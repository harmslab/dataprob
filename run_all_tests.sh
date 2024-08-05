#!/bin/bash

echo "Running flake8"
flake_test=`flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
if [[ "${flake_test}" != 0 ]]; then
    echo "flake failed"
    exit
fi

rm -rf reports
mkdir reports
mkdir reports/junit
mkdir reports/coverage
mkdir reports/badges

echo "Running flake8, aggressive"
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics > reports/flake.txt

echo "Running coverage.py"
coverage erase
coverage run --source ~/miniconda3/lib/python3.12/site-packages/dataprob --branch -m pytest tests/dataprob --runslow --junit-xml=reports/junit/junit.xml

echo "Generating reports"
coverage html
mv htmlcov reports

coverage xml
mv coverage.xml reports/coverage/coverage.xml

genbadge tests
sleep 1
genbadge coverage
sleep 1

#wget https://github.com/harmsm/topiary/actions/workflows/python-app.yml/badge.svg -O ghwf.svg
#wget https://readthedocs.org/projects/topiary-asr/badge/?version=latest -O rtd.svg

mv *.svg docs/badges
