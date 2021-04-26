#!/bin/bash

file=$1

[ -z "$1" ] && file="."

flake8 $file --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 $file --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
