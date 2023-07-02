#!/bin/bash

echo "random"
for i in {1..100}
do
  echo $(python run_game.py random | grep "reward:" | cut -d' ' -f2)
done

echo "model"
for i in {1..100}
do
  echo $(python run_game.py model1687938466.9671688.pt | grep "reward:" | cut -d' ' -f2)
done
