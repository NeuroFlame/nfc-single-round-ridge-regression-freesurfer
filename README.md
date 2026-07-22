### Computation Description

#### Overview
This computation performs a ridge regression on the merged datasets with freesurfer modality from multiple sites using specified covariates and dependent variables. This computation is designed to run within a federated learning environment, where each site performs a local regression analysis, and then global results are aggregated.

The key steps of the algorithm include:

1. **Local Ridge Regression (per site)**:
   - Each site runs ridge regression on its local data, standardizing the covariates and regressing against one or more dependent variables.
   - Statistical metrics (e.g., t-values, p-values, R-squared) are calculated using an ordinary least squares (OLS) ridge regression model to provide interpretability.

2. **Global Aggregation (server)**:
   - After each site computes its local regression results, the server-side computation aggregates the results by averaging the coefficients and other statistics based on the number of subjects (degrees of freedom) per site.

#### Implementation Layout

Computation-specific code lives in `app/code/computation/`. Its declared
workflow is:

1. `fit_local_models` / `aggregate_global_model`
2. `compute_local_metrics` / `aggregate_final_results`
3. `build_output_payloads` at each site

The shared framework owns NVFlare task scheduling, keyed site-result storage,
serialization, state persistence, logging, and output writing. Normal computation
changes should not modify `app/code/framework/` or `app/code/runtime/`, and no
controller, executor, aggregator, or task-name code is required under
`app/code/computation/`.

#### Detailed Steps

1. **Data Preparation**:
   - The computation reads covariate and dependent variable data from CSV files (`covariates.csv` and `data.csv` respectively).
  
2. **Ridge Regression**:
   - The computation fits a ridge regression model using the configured `Lambda` penalty to the standardized covariates and dependent variables.
   - The resulting coefficients are stored for each dependent variable.

3. **OLS Model for Statistical Metrics**:
   - To compute additional statistics (t-values, p-values, R-squared), an OLS model is fitted using the same covariates and dependent variables.
   - The computation extracts these metrics to provide more detailed insights beyond the ridge regression coefficients.

4. **Results**:
   - The aggregated global results are saved as `global_regression_result.json` and the styled `index.html` report and include global and local results:
     - Global and local (per-site) model coefficients
     - Global and local (per-site) t-statistics
     - Global and local (per-site) p-values
     - Global and local (per-site) R-squared
     - Total and local (per-site) degrees of freedom
---

#### Data Format Specification

The computation requires two CSV files as input:

1. **Covariates File (`covariates.csv`)**
2. **Dependent Variables File (`data.csv`)**

Both files must follow a consistent format, though the specific covariates and dependents may vary from study to study based on the `parameters.json` file. The computation expects these files to match the covariate and dependent variable names specified in the `parameters.json` file.

##### Covariates File (`covariates.csv`)

- **Format**: CSV (Comma-Separated Values)
- **Headers**: The file must include a header row where each column name corresponds to a covariate specified in the `parameters.json`.
- **Rows**: Each row represents a subject, where each column contains the value for a specific covariate.
- **Variable Names**: The names of the covariates in the header must match the entries in the `"Covariates"` section of the `parameters.json`.

**General Structure**:
```csv
<Covariate_1>,<Covariate_2>,...,<Covariate_N>
<value_1>,<value_2>,...,<value_N>
<value_1>,<value_2>,...,<value_N>
...
```


##### Dependent Variables File (`data.csv`)

- **Format**: CSV (Comma-Separated Values)
- **Headers**: The file must include a header row where each column name corresponds to a dependent variable specified in the `parameters.json`.
- **Rows**: Each row represents the same subject as in the `covariates.csv`, with values for the dependent variables.
- **Variable Names**: The names of the dependent variables in the header must match the entries in the `"Dependents"` section of the `parameters.json`.

**General Structure**:
```csv
<Dependent_1>,<Dependent_2>,...,<Dependent_N>
<value_1>,<value_2>,...,<value_N>
<value_1>,<value_2>,...,<value_N>
...
```
---

#### Assumptions
- The data provided by each site follows the specified format (standardized covariate and dependent variable headers).
- The computation is run in a federated environment, and each site contributes valid data.

#### Example

- **Input (parameters.json)**:
   ```json   
   {
    "Dependents": {
        "4th-Ventricle":"float",
        "5th-Ventricle":"float"
    },
    "Covariates": {
        "sex":"str",
        "isControl":"bool",
        "age":"float"
    },
    "Lambda": 1,
    "IgnoreSubjectsWithMissingData": false,
    "StrictTypeChecking": false,
    "user_name": "Dr. Jane Smith",
    "user_id": "user-42",
    "log_level": "info"
   }
   ```

`Dependents` and `Covariates` are required. `Lambda`,
`IgnoreSubjectsWithMissingData`, and `StrictTypeChecking` are optional.
`user_name` and `user_id` add report attribution, while `log_level` controls the
site logger. The platform may also supply `site_id_name_map`; when present, its
mapped display names are used in JSON, CSV filenames, and the HTML
report. Site results are keyed by name, so their arrival order does not affect
storage or output labeling.

- **Output files: `global_regression_result.json`, `index.html`, `global_stats.csv`, `local_stats_{site_name}.csv`, and `{site_id}.log`**
- The JSON and HTML files contain global and local results. `global_stats.csv` contains only global results; each `local_stats_{site_name}.csv` contains results for one participating site.
 

#### Output Description
The computation outputs both **site-level** and **global-level** results, which include:
- **Coefficients**: Ridge regression coefficients for each covariate.
- **t-Statistics**: Statistical significance for each coefficient.
- **P-Values**: Probability values indicating significance.
- **R-Squared**: The proportion of variance explained by the model.
- **Degrees of Freedom**: The degrees of freedom used in the regression.
- **Sum of Squared Errors (SSE)**: A measure of the model’s error.

#### Running this computation
To locally run this computation, please clone this repo and run:

```bash
./run_local_simulation.sh site1,site2,site3,site4
```

This script will:
- build the local dev image
- create the NVFlare job
- run the simulator
- print the generated output files under `test_output/simulate_job/<site>/`

For Python-only changes, add `--no-build`. The repository is mounted into the
container, so this still tests current source. Rebuild after changing the
Dockerfile or dependencies.

To compare a single-site run against a local reference bundle:

```bash
./run_local_simulation.sh site1 --compare-bundle ./site1.tgz
```

For lower-level simulator debugging, enter the development container with
`./dockerRun.sh`, then run the same job-building and simulator commands used by
the wrapper:

```bash
python makeJob.py site1,site2,site3,site4
python debugger.py ./job -w ./simulator_workspace -n 4 -c site1,site2,site3,site4
```

#### Debug this computation using IDE
To debug `debugger.py` from an IDE, use:

- **Parameters:** `./job -w ./simulator_workspace -n 1 -c site1`
- **Environment:**
  `PYTHONUNBUFFERED=1;PYTHONPATH=<repository-path>/app/code`

Create the job with `python makeJob.py site1` before starting the debugger.
