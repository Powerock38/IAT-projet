# IAT-projet

Based on https://github.com/aurelienDelageInsaLyon/IAT-projet

## Install
```bash
pip3 install -r requirements.txt
```

## Train
Train a new model (hyperparameters in run_game.py)
```bash
python3 run_game.py
```

## Run
Run with an existing model
```bash
python3 run_game.py <model file>
```

Run with random agent
```bash
python3 run_game.py random
```

## Profile CPU performance
Start profile
```bash
python3 -m cProfile -o run_game.prof run_game.py
```

View profile
```bash
snakeviz run_game.prof
```