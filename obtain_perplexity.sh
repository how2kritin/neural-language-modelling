#!/bin/bash

N=(3 5)
lm_type=("f" "r" "l")
corpora=("corpus/Pride and Prejudice - Jane Austen.txt" "corpus/Ulysses - James Joyce.txt")

for corpus in "${corpora[@]}"; do
  for n in "${N[@]}"; do
    for type in "${lm_type[@]}"; do
      echo "Running $n $type $corpus"
      python ./src/generate.py "$n" "$type" "$corpus" pe
    done
  done
done
