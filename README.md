# Warehouse Operations Optimisation Suite

End-to-end supply chain analytics system built on 10,000 sales records.

## Modules
1. **Demand Forecasting** — Pure-NumPy ARIMA (grid search) + LSTM, Durbin-Watson test
2. **Inventory Optimisation** — EOQ, Safety Stock, Buffer Stock, ROP, GMROI
3. **Warehouse Layout** — Genetic Algorithm on SLP baseline + Monte-Carlo simulation
4. **Order Picking** — ABC Pareto classification + GA-TSP route optimisation
5. **Routing Algorithms** — Hill Climbing vs Simulated Annealing vs GA comparison
6. **KPI Dashboard** — 7-figure Matplotlib output + interactive HTML dashboard

## How to Run
```bash
pip install numpy pandas matplotlib scipy
python warehouse_optimization_v3.py
```
Open `warehouse.html` in any browser and upload your CSV for the interactive dashboard.

## Output
Seven PNG figures saved to `warehouse_outputs/`.