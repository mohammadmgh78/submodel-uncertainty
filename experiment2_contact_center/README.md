# Experiment 2: digital twin of a contact center (Section 6.3)

The KPI is the expected average waiting time of contact group 2 over the 30-minute epoch that follows each of 18 observed
states. Six submodels are estimated: arrivals (A1, A2) and patience and handle times (PH1, PH2) of the two contact groups,
and the contact-side and expert-side routing (RC, RE).

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Real system: contacts, training day, routing decisions | `generate_true_data.py` | `inputs/true_contacts.csv.gz`, `inputs/training_day.csv`, `inputs/routing_decisions_*.csv` |
| 2. Input submodels: point estimates and B = 8 instances | `fit_input_submodels.py` | `inputs/arrival_instances.pkl`, `inputs/copula_*.pkl` |
| 3. Routing submodels: point estimates and B = 8 instances | `fit_routing_submodels.py` | `models/` |
| 4. Simulations | `run_simulations.py truth`, `baseline`, `design`, `comparison` | `data/` |
| 5. Figures 6, 7 and 8 | `analyze.py` | `results/` |
| 6. LH vs. random designs (response letter) | `analyze_lh_vs_random.py` | `results/lh_vs_random_*.csv` |

The outputs of every step are included, so any step can be run on its own. Steps 1 to 4 write to `data_rerun/` and
`models_refit/` or only check the provided files; they never overwrite them.

| Other file | Content |
|---|---|
| `simulator.py` | Discrete-event simulator. `Simulation` uses routing submodels; `TrueSimulation` uses the true routing rules |
| `common.py` | Designs, simulation inputs, initial states, intervals, search, importance scores |
| `routing_wrapper.py` | Wrapper of the neural-network routing models (needed to load `models/`) |

## Method

1. **Pilot** with (B, S, n0) = (4, 2, 10). The variance components, averaged over the states, give n\* = 51 (frequentist)
   and 67 (Bayesian) for τ = 0.1.
2. **Search** over B and S from (4, 2, n\*): one instance or one stack is left out at a time; the parameter with the larger
   predicted change is increased (B is doubled, S grows by one) until the change in the average 90% CI width is below 0.05
   minutes. B is limited to the 8 instances of each submodel. Final values: (8, 6, 51) and (8, 5, 67).
3. **Intervals** with submodel uncertainty: 5% and 95% quantiles of the configuration means. Without submodel uncertainty:
   90% *t*-intervals from the first n baseline replications, with the same n.
4. **Importance scores** per state with 40 bagged trees, averaged over the states.

**Design.** The stacked Latin hypercube design is built by `common.design_lh`: two-level LH designs over the six submodels
give base rows; each row is assigned to a state and expanded to the instances x, x+2, x+4 and x+6. One stack is one base
row per state, i.e., four configurations with B = 8. The final designs therefore have 24 (frequentist) and 20 (Bayesian)
configurations per state.

## Data

| File | Content |
|---|---|
| `inputs/true_contacts.csv.gz` | Contacts of 1,000 days of the real system |
| `inputs/training_day.csv` | Day 1 of the real system (training data of the input submodels) |
| `inputs/observed_states/` | Customers waiting and in service at the observed states. Their remaining patience and handle times are sampled in `common.initial_state` |
| `inputs/routing_decisions_*.csv` | Recorded routing decisions (training data of the routing submodels) |
| `inputs/arrival_instances.pkl`, `copula_point_estimate.pkl`, `copula_instances.pkl` | Input submodels |
| `inputs/copula_instance_draws.csv.gz` | Records from which the instances in `copula_instances.pkl` were recovered |
| `inputs/point_estimate_contacts.csv.gz` | Contacts of 100 days generated from the point-estimate input submodels |
| `inputs/expert_schedules.json`, `threshold_dataset.csv` | Experts and routing thresholds |
| `models/{freq,bayes}_{contact,expert}_{0..7,point}.pkl` | Routing submodels |
| `data/truth.csv` | KPI of 1,000 replications of the real system per state |
| `data/baseline_*.csv` | KPI of 100 replications per state with the point-estimate submodels |
| `data/design_*.csv` | KPI of the stacked LH design, S ≤ 16, n ≤ 69 |
| `data/reference_*.csv`, `lh_vs_random_*.csv` | Runs of the comparison of LH and random designs |

Columns of the design files: `key` (design point; seeds its random streams), the instance of each submodel (`A1` to
`RE`), `state` (0 to 17), `stack`, `dp` (design point), `rep`, and `wait` (the KPI in minutes). A replication without any
contact of group 2 in the epoch has no row.

## Rerun the simulations

```
python run_simulations.py truth
python run_simulations.py baseline
python run_simulations.py design
python run_simulations.py comparison
```

All runs are seeded per design point and reproduce the files in `data/` up to floating-point rounding. `design` and
`comparison` can be interrupted and resumed.
