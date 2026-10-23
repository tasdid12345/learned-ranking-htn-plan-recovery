# Learned ranking of HTN recovery options

## Motivation
The paper "Plan recovery in reactive htns using symbolic planning" (2015) combines reactive hierarchical task networks with symbolic planning to recover plans when execution fails. This experiment asks whether a learned model can make that recovery cheaper by deciding which recovery option to try first.

## Question
Can a LightGBM model that ranks recovery options reduce the symbolic search effort (node expansions) needed to repair a broken hierarchical plan, compared with trying options in a fixed library order?

## Dataset
No public dataset fits, so 4,000 seeded grid-world episodes are simulated inside `analysis.py`. An HTN-style plan (an L-shaped route to the goal) breaks when a wall of obstacles or locked doors appears. Each episode has 12 candidate recovery methods (waypoint decompositions, one of them fetching a key), and a breadth-first symbolic planner checks each one. The split is 70/30 by episode, and the test split is the same for both models. This is **not** the paper's own dataset.

## Baseline vs improved
- **Baseline:** a stand-in for the standard symbolic recovery. It tries the options in fixed library order, then falls back to a full replan.
- **Improved:** a class-balanced LightGBM, tuned with grouped cross-validation on the training episodes. It predicts which options will succeed from state, failure and option features. It then tries them in ranked order with the same planner, costs, budget and fallback.

Main metric: mean node expansions until a valid recovery is found (lower is better). Secondary metrics: success rate within a fixed expansion budget, and the share of episodes where the first option tried works.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> L Ould Ouali, C Rich, N Sabouret (2015). *Plan recovery in reactive htns using symbolic planning*. International Conference on Artificial. https://link.springer.com/chapter/10.1007/978-3-319-21365-1_33

Paper author: Prof. Charles Rich, Worcester Polytechnic Institute (rich@wpi.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
