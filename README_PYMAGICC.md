# Advanced Climate Model using PyMAGICC

This script provides sophisticated climate modeling using MAGICC6 (Model for the Assessment of Greenhouse Gas Induced Climate Change), the same model used by the IPCC for climate projections.

## Features

- **Professional Climate Modeling**: Uses MAGICC6, the industry-standard climate model
- **Multiple Scenarios**: Compare 25%, 50%, 75%, and 100% emission scenarios
- **Long-term Projections**: Simulations from 2025 to 2300
- **Comprehensive Output**: Temperature, CO2 concentration, radiative forcing, and more
- **Publication-Quality Visualizations**: 9-panel figure with detailed comparisons
- **Detailed Data Export**: CSV file with all results for further analysis

## Requirements

### System Requirements
- Python 3.7 or higher
- Windows, macOS, or Linux
- ~500 MB disk space for MAGICC model files

### Python Packages
```bash
pip install pymagicc matplotlib numpy pandas
```

## Installation

### Step 1: Install Python
If you don't have Python installed, download it from [python.org](https://www.python.org/downloads/)

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv climate_env

# Activate it
# On Windows:
climate_env\Scripts\activate
# On macOS/Linux:
source climate_env/bin/activate
```

### Step 3: Install Required Packages
```bash
pip install pymagicc matplotlib numpy pandas
```

**Note**: The first time you run the script, pymagicc will automatically download the MAGICC model files (~200 MB). This is a one-time download.

### Step 4: Download the Script
Save `advanced_climate_model_pymagicc.py` to your working directory.

## Usage

### Basic Usage
```bash
python advanced_climate_model_pymagicc.py
```

The script will:
1. Create emission scenarios (25%, 50%, 75%, 100% of 2025 baseline)
2. Run MAGICC6 simulations (takes 2-5 minutes)
3. Generate comprehensive visualizations
4. Save detailed results to CSV

### Expected Runtime
- First run (with MAGICC download): 3-10 minutes
- Subsequent runs: 2-5 minutes
- The script will show progress for each scenario

## Output Files

### 1. magicc_climate_scenarios.png
Comprehensive 9-panel visualization including:
- Temperature projections for all scenarios
- CO2 concentration trajectories
- Radiative forcing
- Comparison bars for 2100 temperatures
- Temperature change rates
- And more!

### 2. magicc_results_detailed.csv
Detailed year-by-year data containing:
- Temperature anomalies (°C)
- CO2 concentrations (ppm)
- Radiative forcing (W/m²)
- For each scenario and the baseline

## Customization

### Modify Emission Scenarios
Edit the `main()` function to change scenarios:

```python
# Change reduction levels
reduction_levels = [0.20, 0.40, 0.60, 0.80]  # 20%, 40%, 60%, 80%

# Change target year
trough_year = 2040  # Reach target by 2040 instead of 2050
```

### Extend Simulation Period
Modify the `create_emission_scenarios()` call:

```python
scenarios, baseline_2025 = model.create_emission_scenarios(
    reduction_levels=reduction_levels,
    trough_year=2050,
    start_year=2025,
    end_year=2400  # Extend to 2400
)
```

### Change Baseline Scenario
In the `__init__` method of `AdvancedClimateModel`:

```python
# Use different RCP baseline
from pymagicc.scenarios import rcp26, rcp45, rcp60, rcp85

self.base_scenario = rcp26  # More aggressive baseline
# or
self.base_scenario = rcp85  # Higher emissions baseline
```

## Understanding the Results

### Temperature Anomalies
- Measured relative to pre-industrial levels (1850-1900)
- Paris Agreement targets: 1.5°C (target) and 2.0°C (limit)

### CO2 Concentrations
- Pre-industrial: ~280 ppm
- Current (2024): ~420 ppm
- Results show both immediate and long-term stabilization

### Emission Scenarios
- **25% scenario**: 75% reduction from 2025 baseline
- **50% scenario**: 50% reduction from 2025 baseline
- **75% scenario**: 25% reduction from 2025 baseline
- **100% scenario**: No reduction (flat at current levels)

## Troubleshooting

### "ModuleNotFoundError: No module named 'pymagicc'"
```bash
pip install pymagicc
```

### "MAGICC6 binary not found"
The first run will automatically download MAGICC. If this fails:
```python
from pymagicc import MAGICC6
MAGICC6.download()
```

### Script runs slowly
MAGICC is computationally intensive. Expected runtime is 2-5 minutes for 4-5 scenarios. Consider:
- Reducing the number of scenarios
- Shortening the simulation period
- Running on a faster computer

### Memory errors
MAGICC can use significant memory. Try:
- Closing other applications
- Reducing the time resolution (if you modify the code)
- Running scenarios one at a time

## Scientific Background

### About MAGICC
- Developed by climate scientists at NCAR and UCAR
- Used by IPCC Assessment Reports
- Includes sophisticated carbon cycle modeling
- Accounts for climate feedbacks and ocean heat uptake
- Models multiple greenhouse gases and aerosols

### Model Limitations
- Simplified representation of Earth system
- Does not include tipping points or abrupt changes
- Assumes smooth emission transitions
- Does not model regional climate variations

### References
- Meinshausen et al. (2011): "The RCP greenhouse gas concentrations and their extensions from 1765 to 2300"
- IPCC AR5 and AR6 Assessment Reports
- MAGICC documentation: http://www.magicc.org/

## Advanced Usage

### Running Individual Scenarios

```python
from pymagicc import MAGICC6
from pymagicc.scenarios import rcp45

# Modify a scenario
scenario = rcp45.copy()
# ... make modifications ...

# Run MAGICC
with MAGICC6() as magicc:
    results = magicc.run(scenario)

# Extract results
temperature = results["SURFACE_TEMP"]["WORLD"]["Annual Mean"]
```

### Accessing More Variables

MAGICC outputs many variables. Common ones:
- `SURFACE_TEMP`: Surface temperature
- `CO2_CONC`: CO2 concentration
- `TOTAL_INCLVOLCANIC_RF`: Total radiative forcing
- `OCEAN_TEMP`: Ocean temperature
- `SEA_LEVEL_RISE`: Sea level rise
- `CH4_CONC`: Methane concentration

Access them like:
```python
sea_level = results["SEA_LEVEL_RISE"]["WORLD"]["Annual Mean"]
```

## Support and Contribution

### Issues
If you encounter issues:
1. Check that all dependencies are installed
2. Verify Python version (3.7+)
3. Ensure adequate disk space and memory
4. Check the pymagicc documentation: https://github.com/openscm/pymagicc

### Citation
If you use this script in research or publications, please cite:
- MAGICC: Meinshausen et al. (2011)
- PyMAGICC: Gieseke et al. (2018)

## License

This script is provided as-is for educational and research purposes. MAGICC itself has its own licensing terms - see the pymagicc documentation for details.

## Version History

- v1.0 (2025): Initial release
  - Support for multiple emission scenarios
  - Comprehensive visualization suite
  - Extended projections to 2300

---

**Note**: Climate modeling involves significant scientific uncertainty. These results should be interpreted as projections based on current understanding, not exact predictions. Always consult with climate scientists for policy-relevant interpretations.
