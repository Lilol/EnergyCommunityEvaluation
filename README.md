# Energy Community Evaluation

A framework for evaluating energy communities using household consumption, PV generation, tariffs, and batteries.

---

## How to Cite

If you use this framework in your research, please cite it as:

```bibtex
@software{energy_community_evaluation,
  author = {Barancsuk, Lilla},
  title = {Energy Community Evaluation Framework},
  year = {2025},
  url = {https://github.com/Lilol/EnergyCommunityEvaluation},
  note = {A framework for evaluating renewable energy communities with parametric analysis capabilities}
}
```

Alternatively, you can cite it in text as:
> Barancsuk, L. (2025). Energy Community Evaluation Framework. https://github.com/Lilol/EnergyCommunityEvaluation

---

## Table of Contents

- [How to Cite](#how-to-cite)
- [Features](#features)
- [Google Colab Demo](#google-colab-demo)
- [Setup Guide](#setup-guide)
  - [Preprocessing](#preprocessing)
  - [Parametric Evaluation](#parametric-evaluation)
  - [Renewable Energy Community Evaluation](#renewable-energy-community-evaluation)
- [Contact](#contact)

---

## Features

- **Data Preprocessing**: Automated processing of household consumption, PV generation, and billing data
- **Parametric Analysis**: Comprehensive evaluation of energy communities with varying parameters
- **Battery Simulation**: Battery Energy Storage System (BESS) modeling and optimization
- **Multiple Metrics**: Physical, economic, and environmental impact assessments
- **Visualization**: Built-in plotting and analysis tools
- **Flexible Configuration**: Easy-to-use configuration file system

---

## Google Colab Demo

A student-friendly Colab workflow is available in the `colab/` folder.

- Notebook: `colab/EnergyCommunityEvaluation_Student_Demo.ipynb`
- Helper utilities: `colab/demo_setup.py`
- Notes and quick-start guide: `colab/README.md`

The notebook creates a small synthetic municipality called `ColabTown`, then runs the real `preprocessing.py` and `run_parametric_evaluation.py` entry points so students can explore the full workflow without preparing a large municipal dataset first.

If this repository is hosted on GitHub, you can open the notebook in Colab with:

```text
https://colab.research.google.com/github/Lilol/EnergyCommunityEvaluation/blob/main/colab/EnergyCommunityEvaluation_Student_Demo.ipynb
```

---

## Setup Guide

### Preprocessing

#### What `preprocessing.py` Does:

1. **Reads and processes input data** — household data, PV generation, tariffs, bills.
2. **Validates and transforms** the data — cleaning and converting to `xarray.DataArray` structures.
3. **Stores processed data** in structured `xarray` format.
4. **Generates visualizations** for load and generation.
5. **Writes outputs** — including cleaned load/generation/user data as `.csv` and `.nc` files.

> This script automates preprocessing for multiple families and users.

####  How to Run `preprocessing.py`

1. Clone the repo to your project root: `<root>`
2. Copy the default config:
   ```
   cp <root>/config/example_config.ini <root>/config/config.ini
   ```
3. Set the input/output file paths in `config.ini`:
   ```ini
   [path]
   root=path/to/files
   ```

4. Ensure this folder structure under the root:

   ```
   input/
   ├── common/
   │   ├── arera.csv
   │   └── y_ref_gse.csv
   ├── DatabaseGSE/
   │   └── gse_ref_profiles.csv
   └── DatiComuni/
       └── SettimoTorinese/
           ├── PVGIS/ <generator_production_files>
           ├── PVSOL/ <generator_production_files>
           ├── bollette_domestici.csv
           ├── dati_bollette.csv
           ├── lista_impianti.csv
           └── lista_pod.csv
   ```

5. Outputs will be created at:
   ```
   output/DatiProcessati/SettimoTorinese/
   ├── Loads/ <loads_by_ids>
   ├── Generators/ <generators_by_ids>
   ├── data_users.csv / data_plants.csv / ...
   └── families_<num_families>.csv
   ```

6. Set up your Python environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

7. Run the script:
   ```bash
   python preprocessing.py
   ```

---

### Parametric Evaluation

#### What `run_parametric_evaluation.py` Does:

1. **Creates datasets** in `xarray` format:
   - Aggregates monthly and ToU (time-of-use) data
   - Outputs:
     - `tou_months`: ToU-separated monthly data
     - `energy_year`: Yearly energy data

2. **Computes metrics**:
   - Physical, environmental, economic metrics
   - Simulates battery energy storage systems (BESS)

3. **Visualizes** the evaluation process.

4. **Exports results** to `.csv`

#### Example result figures

Difference plot:

![Difference Plot](images/difference.png)

Main metrics overview:

![Metrics Plot](images/metrics.png)

Additional metrics comparison:

![Metrics Plot 2](images/metrics2.png)

Time-aggregation view:

![Time Aggregation Plot](images/time_aggregation.png)

#### How to Run `run_parametric_evaluation.py`

1. Clone the repo to your project root: `<root>`
2. Copy the config:
   ```
   cp <root>/config/example_config.ini <root>/config/config.ini
   ```

3. Set paths in `config.ini`:
   ```ini
   [path]
   root=path/to/files
   ```

4. Define parameters in `[parametric_evaluation]`:
   - Option 1: Cartesian product of battery sizes × families:
     ```ini
     evaluation_parameters = {'bess_sizes': [1, 2], 'number_of_families': [20, 50, 70]}
     ```
   - Option 2: Specific family sets for each battery:
     ```ini
     evaluation_parameters = {'bess_sizes': {0: [2, 3, 4], 1: [20, 24]}}
     ```
   - Option 3: Battery sets for each number of families:
     ```ini
     evaluation_parameters = {'number_of_families': {20: [10, 20, 40], 50: [60, 9]}}
     ```

5. Choose metrics under `to_evaluate`:
   ```ini
   to_evaluate = ['physical', 'economic', 'environmental', 'all']
   ```
   Options include:
   - `physical`: self-consumption, self-sufficiency
   - `economic`: capex, opex
   - `environmental`: baseline emissions, emissions in case of REC establishment, savings
   - `metric_targets`, `time_aggregation`, `all`

6. Define cost/emissions in:
   ```
   data/cost_of_equipment.csv
   data/emission_factors.csv
   ```

7. Set up the environment and run:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python parametric_evaluation/run_parametric_evaluation.py
   ```

---

### Renewable Energy Community Evaluation

*Coming soon: Detailed evaluation of renewable energy community configurations and performance metrics.*

---

## Contact

**Maintained by**: Lilla Barancsuk  
**GitHub**: https://github.com/Lilol  
**Issues**: Please report bugs and feature requests via [GitHub Issues](https://github.com/Lilol/EnergyCommunityEvaluation/issues)

---

## License

See [LICENSE](LICENSE) file for details.
