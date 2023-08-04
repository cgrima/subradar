#!/bin/bash -e


COV=coverage3

$COV run ./test_subradar1.py

echo "Run $COV report -m to see results"
