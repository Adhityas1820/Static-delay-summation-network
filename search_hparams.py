import subprocess
import itertools
import re
import sys

layers = [1, 2, 3, 4]
hiddens = [32, 64]
epochs = [20, 30, 40]

best_f1 = -1
best_mae = 1e9
best_params = None

results = []

print("Starting Hyperparameter Grid Search...", flush=True)

for l, h, e in itertools.product(layers, hiddens, epochs):
    print(f"\n--- Testing layers={l}, hidden={h}, epochs={e} ---", flush=True)
    
    cmd1 = f"python -m dash_code.train --epochs {e} --hidden {h} --layers {l}"
    print(f"Running: {cmd1}", flush=True)
    subprocess.run(cmd1, shell=True, check=True)
    
    cmd2 = f"python stage4.py --no-overlay"
    print(f"Running: {cmd2}", flush=True)
    subprocess.run(cmd2, shell=True, check=True)
    
    with open("output.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    match = re.search(r"count MAE ([\d\.]+)  \|  timing F1 ([\d\.]+)", content)
    if match:
        mae = float(match.group(1))
        f1 = float(match.group(2))
        
        results.append((l, h, e, mae, f1))
        print(f"Result: MAE={mae}, F1={f1}", flush=True)
        
        if f1 > best_f1 or (f1 == best_f1 and mae < best_mae):
            best_f1 = f1
            best_mae = mae
            best_params = (l, h, e)
            print(f"** NEW BEST **", flush=True)
    else:
        print("Could not parse output.txt metrics.", flush=True)

print("\n--- GRID SEARCH COMPLETE ---")
print(f"Best Params: layers={best_params[0]}, hidden={best_params[1]}, epochs={best_params[2]}")
print(f"Best Score : F1={best_f1}, MAE={best_mae}")

with open("search_results.txt", "w", encoding="utf-8") as f:
    f.write("Grid Search Results:\n")
    for res in results:
        f.write(f"layers={res[0]}, hidden={res[1]}, epochs={res[2]} -> MAE={res[3]}, F1={res[4]}\n")
    f.write(f"\nBest configuration: layers={best_params[0]}, hidden={best_params[1]}, epochs={best_params[2]}\n")
    f.write(f"Best Score: F1={best_f1}, MAE={best_mae}\n")
