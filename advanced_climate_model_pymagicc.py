"""
Advanced Climate Model using PyMAGICC
======================================
This script uses the MAGICC6 climate model to simulate multiple emission scenarios.
MAGICC (Model for the Assessment of Greenhouse Gas Induced Climate Change) is used
by the IPCC and provides more sophisticated climate modeling than simple energy balance models.

Requirements:
    pip install pymagicc matplotlib numpy pandas

Usage:
    python advanced_climate_model_pymagicc.py

Author: Climate Modeling Script
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pymagicc import MAGICC6
from pymagicc.scenarios import rcp26, rcp45, rcp60, rcp85
import warnings
warnings.filterwarnings('ignore')


class AdvancedClimateModel:
    """
    Advanced climate modeling using MAGICC6
    """
    
    def __init__(self):
        """Initialize the MAGICC6 model"""
        self.magicc = None
        self.base_scenario = rcp45  # Use RCP4.5 as baseline scenario
        
    def create_emission_scenarios(self, reduction_levels=[0.25, 0.50, 0.75, 1.00],
                                  trough_year=2050, start_year=2025, end_year=2300):
        """
        Create custom emission scenarios

        Parameters:
        -----------
        reduction_levels : list
            Emission levels as fraction of 2025 baseline (e.g., 0.25 = 25% of baseline)
        trough_year : int
            Year when emission reduction target is reached
        start_year : int
            Starting year for scenarios
        end_year : int
            Ending year for scenarios

        Returns:
        --------
        scenarios : dict
            Dictionary of scenario DataFrames ready for MAGICC
        """
        import datetime as dt
        scenarios = {}

        # Get baseline emissions from RCP scenario
        baseline_scenario = self.base_scenario.copy()

        # RCP scenarios use different metadata structure - need to filter properly
        # Extract CO2 emissions - MAGICC uses two components: Fossil+Industrial and AFOLU
        # Filter for World region only (global total)
        co2_fossil = baseline_scenario.filter(variable="Emissions|CO2|MAGICC Fossil and Industrial",
                                             region="World")
        co2_afolu = baseline_scenario.filter(variable="Emissions|CO2|MAGICC AFOLU",
                                            region="World")

        # Get the timeseries data
        co2_fossil_ts = co2_fossil.timeseries()
        co2_afolu_ts = co2_afolu.timeseries()

        # Get 2025 baseline value - columns are datetime objects
        dt_2025 = dt.datetime(2025, 1, 1)
        if dt_2025 in co2_fossil_ts.columns:
            baseline_2025_fossil = co2_fossil_ts[dt_2025].iloc[0]
            baseline_2025_afolu = co2_afolu_ts[dt_2025].iloc[0]
        else:
            # Find nearest year
            dt_2020 = dt.datetime(2020, 1, 1)
            dt_2030 = dt.datetime(2030, 1, 1)
            # Interpolate between 2020 and 2030
            if dt_2020 in co2_fossil_ts.columns and dt_2030 in co2_fossil_ts.columns:
                fossil_2020 = co2_fossil_ts[dt_2020].iloc[0]
                fossil_2030 = co2_fossil_ts[dt_2030].iloc[0]
                baseline_2025_fossil = fossil_2020 + (fossil_2030 - fossil_2020) * 0.5

                afolu_2020 = co2_afolu_ts[dt_2020].iloc[0]
                afolu_2030 = co2_afolu_ts[dt_2030].iloc[0]
                baseline_2025_afolu = afolu_2020 + (afolu_2030 - afolu_2020) * 0.5
            else:
                print("Warning: Could not find 2025 or nearby years")
                return {}, 0

        baseline_2025 = baseline_2025_fossil + baseline_2025_afolu

        print(f"2025 Baseline CO2 emissions: {baseline_2025:.2f} GtC/yr")
        print(f"  Fossil & Industrial: {baseline_2025_fossil:.2f} GtC/yr")
        print(f"  AFOLU: {baseline_2025_afolu:.2f} GtC/yr")
        print(f"(approximately {baseline_2025 * 3.67:.1f} GtCO2/yr total)")
        print()

        # Create scenarios for each reduction level
        for level in reduction_levels:
            scenario = baseline_scenario.copy()
            target_emission = baseline_2025 * level

            # Get the CO2 emissions data for modification - both components
            scenario_co2_fossil = scenario.filter(variable="Emissions|CO2|MAGICC Fossil and Industrial",
                                                 region="World")
            scenario_co2_afolu = scenario.filter(variable="Emissions|CO2|MAGICC AFOLU",
                                                region="World")

            # Get timeseries for both components
            fossil_ts = scenario_co2_fossil.timeseries().copy()
            afolu_ts = scenario_co2_afolu.timeseries().copy()

            # Get available years as datetime columns
            for col in fossil_ts.columns:
                if isinstance(col, dt.datetime):
                    year = col.year

                    if year < start_year:
                        # Keep original RCP values before start
                        continue
                    elif year <= trough_year:
                        # Linear transition to target
                        if year == start_year:
                            new_emission = baseline_2025
                        else:
                            progress = (year - start_year) / (trough_year - start_year)
                            new_emission = baseline_2025 - (baseline_2025 - target_emission) * progress

                        # Scale both components proportionally to reach new_emission
                        current_fossil = fossil_ts.iloc[0, fossil_ts.columns.get_loc(col)]
                        current_afolu = afolu_ts.iloc[0, afolu_ts.columns.get_loc(col)]
                        current_total = current_fossil + current_afolu

                        if current_total > 0:
                            scale_factor = new_emission / current_total
                            fossil_ts.iloc[0, fossil_ts.columns.get_loc(col)] = current_fossil * scale_factor
                            afolu_ts.iloc[0, afolu_ts.columns.get_loc(col)] = current_afolu * scale_factor
                    else:
                        # Maintain target level - scale proportionally
                        current_fossil = fossil_ts.iloc[0, fossil_ts.columns.get_loc(col)]
                        current_afolu = afolu_ts.iloc[0, afolu_ts.columns.get_loc(col)]
                        current_total = current_fossil + current_afolu

                        if current_total > 0:
                            scale_factor = target_emission / current_total
                            fossil_ts.iloc[0, fossil_ts.columns.get_loc(col)] = current_fossil * scale_factor
                            afolu_ts.iloc[0, afolu_ts.columns.get_loc(col)] = current_afolu * scale_factor

            # Update the scenario with the modified emissions
            # Remove World region data for CO2 variables
            scenario_filtered = scenario.filter(variable="Emissions|CO2*", region="World", keep=False)

            # Add back the modified data
            scenario_filtered = scenario_filtered.append(scenario_co2_fossil.__class__(fossil_ts))
            scenario_filtered = scenario_filtered.append(scenario_co2_afolu.__class__(afolu_ts))

            scenarios[level] = scenario_filtered

        return scenarios, baseline_2025
    
    def run_scenarios(self, scenarios):
        """
        Run MAGICC for multiple scenarios
        
        Parameters:
        -----------
        scenarios : dict
            Dictionary of scenarios to run
            
        Returns:
        --------
        results : dict
            Dictionary of MAGICC results for each scenario
        """
        results = {}
        
        with MAGICC6() as magicc:
            print("Running MAGICC6 simulations...")
            print("(This may take a few minutes...)")
            print()
            
            for i, (level, scenario) in enumerate(scenarios.items()):
                pct = int(level * 100)
                print(f"  [{i+1}/{len(scenarios)}] Running {pct}% emission scenario...")
                
                # Run MAGICC with this scenario
                result = magicc.run(scenario)
                results[level] = result
        
        print("✓ All simulations complete!")
        print()
        return results
    
    def extract_results(self, results, baseline_result=None):
        """
        Extract key climate variables from MAGICC results
        
        Parameters:
        -----------
        results : dict
            Dictionary of MAGICC results
        baseline_result : MAGICCData, optional
            Baseline scenario results
            
        Returns:
        --------
        extracted : dict
            Dictionary containing temperature, CO2, and other key variables
        """
        extracted = {
            'temperature': {},
            'co2_concentration': {},
            'forcing': {},
            'emissions': {}
        }
        
        for level, result in results.items():
            # Extract surface temperature - use filter to get data
            temp_data = result.filter(variable="Surface Temperature")
            if len(temp_data) == 0:
                # Try alternative variable names
                temp_data = result.filter(variable="*Temperature*")
            temp = temp_data.timeseries().iloc[0]
            extracted['temperature'][level] = temp
            
            # Extract atmospheric CO2 concentration
            co2_data = result.filter(variable="Atmospheric Concentrations|CO2")
            if len(co2_data) == 0:
                co2_data = result.filter(variable="*CO2*Concentration*")
            co2 = co2_data.timeseries().iloc[0]
            extracted['co2_concentration'][level] = co2
            
            # Extract radiative forcing
            forcing_data = result.filter(variable="Radiative Forcing")
            if len(forcing_data) == 0:
                forcing_data = result.filter(variable="*Forcing*")
            forcing = forcing_data.timeseries().iloc[0]
            extracted['forcing'][level] = forcing
            
        # Add baseline if provided
        if baseline_result is not None:
            temp_data = baseline_result.filter(variable="Surface Temperature")
            if len(temp_data) == 0:
                temp_data = baseline_result.filter(variable="*Temperature*")
            extracted['temperature']['baseline'] = temp_data.timeseries().iloc[0]
            
            co2_data = baseline_result.filter(variable="Atmospheric Concentrations|CO2")
            if len(co2_data) == 0:
                co2_data = baseline_result.filter(variable="*CO2*Concentration*")
            extracted['co2_concentration']['baseline'] = co2_data.timeseries().iloc[0]
            
            forcing_data = baseline_result.filter(variable="Radiative Forcing")
            if len(forcing_data) == 0:
                forcing_data = baseline_result.filter(variable="*Forcing*")
            extracted['forcing']['baseline'] = forcing_data.timeseries().iloc[0]
        
        return extracted
    
    def print_summary(self, extracted, reduction_levels):
        """Print summary statistics for all scenarios"""
        
        print("=" * 80)
        print("SIMULATION RESULTS SUMMARY")
        print("=" * 80)
        print()
        
        for level in reduction_levels:
            temp = extracted['temperature'][level]
            co2 = extracted['co2_concentration'][level]
            
            print(f"{'─' * 80}")
            print(f"SCENARIO: {int((1-level)*100)}% REDUCTION TO {int(level*100)}% BY 2050")
            print(f"{'─' * 80}")
            print()
            
            # Key years
            years_to_report = [2050, 2100, 2200, 2300]
            
            print("Temperature (°C above pre-industrial):")
            for year in years_to_report:
                if year in temp.index:
                    print(f"  {year}: {temp.loc[year]:.2f}°C")
            print(f"  Peak: {temp.max():.2f}°C (Year {temp.idxmax()})")
            print()
            
            print("CO2 Concentration (ppm):")
            for year in years_to_report:
                if year in co2.index:
                    print(f"  {year}: {co2.loc[year]:.1f} ppm")
            print()
            
            # Paris Agreement assessment
            if 2100 in temp.index:
                temp_2100 = temp.loc[2100]
                if temp_2100 < 1.5:
                    status = "✓ Stays below 1.5°C Paris target"
                elif temp_2100 < 2.0:
                    status = "⚠ Exceeds 1.5°C but stays below 2.0°C Paris limit"
                else:
                    status = "✗ Exceeds 2.0°C Paris limit"
                print(f"Paris Agreement Status (2100): {status}")
            print()
        
        # Baseline comparison
        if 'baseline' in extracted['temperature']:
            print("=" * 80)
            print("BASELINE SCENARIO (RCP4.5 - No additional action)")
            print("=" * 80)
            temp_base = extracted['temperature']['baseline']
            co2_base = extracted['co2_concentration']['baseline']
            
            for year in years_to_report:
                if year in temp_base.index:
                    print(f"  {year}: {temp_base.loc[year]:.2f}°C, {co2_base.loc[year]:.1f} ppm CO2")
            print()


def create_comprehensive_plots(extracted, reduction_levels, output_file='magicc_climate_scenarios.png'):
    """
    Create comprehensive visualization of all scenarios
    
    Parameters:
    -----------
    extracted : dict
        Extracted climate variables
    reduction_levels : list
        List of reduction levels
    output_file : str
        Output filename for the plot
    """
    
    # Color scheme
    colors = {
        0.25: '#2E86AB',
        0.50: '#A23B72',
        0.75: '#F18F01',
        1.00: '#C73E1D',
        'baseline': '#666666'
    }
    
    labels = {
        0.25: '25% (75% reduction)',
        0.50: '50% (50% reduction)',
        0.75: '75% (25% reduction)',
        1.00: '100% (no reduction)',
        'baseline': 'RCP4.5 Baseline'
    }
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # Plot 1: Temperature trajectories (all scenarios)
    ax1 = fig.add_subplot(gs[0, :])
    
    if 'baseline' in extracted['temperature']:
        temp_base = extracted['temperature']['baseline']
        ax1.plot(temp_base.index, temp_base.values, '--', linewidth=2, 
                color=colors['baseline'], label=labels['baseline'], alpha=0.7, zorder=1)
    
    for level in reduction_levels:
        temp = extracted['temperature'][level]
        ax1.plot(temp.index, temp.values, '-', linewidth=2.5,
                color=colors[level], label=labels[level], zorder=2)
    
    ax1.axhline(y=1.5, color='orange', linestyle='--', alpha=0.6, linewidth=1.5, label='1.5°C Paris Target')
    ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='2.0°C Paris Limit')
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Temperature Anomaly (°C above pre-industrial)', fontsize=12)
    ax1.set_title('Global Temperature Projections - MAGICC6 Model', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2020, 2300)
    
    # Plot 2: CO2 Concentration
    ax2 = fig.add_subplot(gs[1, 0])
    
    if 'baseline' in extracted['co2_concentration']:
        co2_base = extracted['co2_concentration']['baseline']
        ax2.plot(co2_base.index, co2_base.values, '--', linewidth=2,
                color=colors['baseline'], label=labels['baseline'], alpha=0.7, zorder=1)
    
    for level in reduction_levels:
        co2 = extracted['co2_concentration'][level]
        ax2.plot(co2.index, co2.values, '-', linewidth=2,
                color=colors[level], label=labels[level], zorder=2)
    
    ax2.axhline(y=280, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='Pre-industrial')
    ax2.axhline(y=420, color='orange', linestyle=':', alpha=0.5, linewidth=1, label='Current (~2024)')
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Atmospheric CO₂ (ppm)', fontsize=11)
    ax2.set_title('CO₂ Concentration', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2020, 2300)
    
    # Plot 3: Radiative Forcing
    ax3 = fig.add_subplot(gs[1, 1])
    
    if 'baseline' in extracted['forcing']:
        forcing_base = extracted['forcing']['baseline']
        ax3.plot(forcing_base.index, forcing_base.values, '--', linewidth=2,
                color=colors['baseline'], label=labels['baseline'], alpha=0.7, zorder=1)
    
    for level in reduction_levels:
        forcing = extracted['forcing'][level]
        ax3.plot(forcing.index, forcing.values, '-', linewidth=2,
                color=colors[level], label=labels[level], zorder=2)
    
    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('Radiative Forcing (W/m²)', fontsize=11)
    ax3.set_title('Total Radiative Forcing', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(2020, 2300)
    
    # Plot 4: Temperature in 2100
    ax4 = fig.add_subplot(gs[1, 2])
    
    temps_2100 = []
    scenario_labels = []
    bar_colors = []
    
    for level in reduction_levels:
        temp = extracted['temperature'][level]
        if 2100 in temp.index:
            temps_2100.append(temp.loc[2100])
            scenario_labels.append(f"{int(level*100)}%")
            bar_colors.append(colors[level])
    
    if 'baseline' in extracted['temperature'] and 2100 in extracted['temperature']['baseline'].index:
        temps_2100.append(extracted['temperature']['baseline'].loc[2100])
        scenario_labels.append("Baseline")
        bar_colors.append(colors['baseline'])
    
    bars = ax4.bar(range(len(temps_2100)), temps_2100, color=bar_colors, alpha=0.8, edgecolor='black')
    ax4.axhline(y=1.5, color='orange', linestyle='--', alpha=0.6, linewidth=2, label='1.5°C Target')
    ax4.axhline(y=2.0, color='red', linestyle='--', alpha=0.6, linewidth=2, label='2.0°C Limit')
    ax4.set_xticks(range(len(scenario_labels)))
    ax4.set_xticklabels(scenario_labels, rotation=45, ha='right')
    ax4.set_ylabel('Temperature (°C)', fontsize=11)
    ax4.set_title('Temperature in 2100 by Scenario', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, temps_2100)):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}°C',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 5: Temperature trajectories (zoomed on reduction scenarios)
    ax5 = fig.add_subplot(gs[2, 0])
    
    for level in reduction_levels:
        temp = extracted['temperature'][level]
        ax5.plot(temp.index, temp.values, '-', linewidth=2.5,
                color=colors[level], label=labels[level])
    
    ax5.axhline(y=1.5, color='orange', linestyle='--', alpha=0.6, linewidth=1.5)
    ax5.axhline(y=2.0, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax5.set_xlabel('Year', fontsize=11)
    ax5.set_ylabel('Temperature Anomaly (°C)', fontsize=11)
    ax5.set_title('Temperature: Reduction Scenarios (Zoomed)', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper left', fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(2020, 2300)
    ax5.set_ylim(0, 4.5)
    
    # Plot 6: Warming rate (decade-average change)
    ax6 = fig.add_subplot(gs[2, 1])
    
    for level in reduction_levels:
        temp = extracted['temperature'][level]
        # Calculate 10-year moving rate of change
        temp_smooth = temp.rolling(window=10, center=True).mean()
        rate = temp_smooth.diff(10) * 1  # Change per decade
        ax6.plot(rate.index, rate.values, '-', linewidth=2,
                color=colors[level], label=labels[level], alpha=0.8)
    
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax6.set_xlabel('Year', fontsize=11)
    ax6.set_ylabel('Warming Rate (°C/decade)', fontsize=11)
    ax6.set_title('Rate of Temperature Change', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(2020, 2300)
    
    # Plot 7: CO2 concentration in 2100 and 2300
    ax7 = fig.add_subplot(gs[2, 2])
    
    x_pos = np.arange(len(reduction_levels))
    width = 0.35
    
    co2_2100 = []
    co2_2300 = []
    
    for level in reduction_levels:
        co2 = extracted['co2_concentration'][level]
        co2_2100.append(co2.loc[2100] if 2100 in co2.index else np.nan)
        co2_2300.append(co2.loc[2300] if 2300 in co2.index else np.nan)
    
    bars1 = ax7.bar(x_pos - width/2, co2_2100, width, label='2100', alpha=0.8, edgecolor='black')
    bars2 = ax7.bar(x_pos + width/2, co2_2300, width, label='2300', alpha=0.8, edgecolor='black')
    
    # Color bars by scenario
    for i, level in enumerate(reduction_levels):
        bars1[i].set_color(colors[level])
        bars2[i].set_color(colors[level])
        bars2[i].set_alpha(0.6)
    
    ax7.set_xlabel('Emission Scenario', fontsize=11)
    ax7.set_ylabel('CO₂ Concentration (ppm)', fontsize=11)
    ax7.set_title('CO₂ Levels: 2100 vs 2300', fontsize=12, fontweight='bold')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([f"{int(level*100)}%" for level in reduction_levels])
    ax7.legend(loc='upper left', fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add overall title
    fig.suptitle('Climate Scenario Analysis using MAGICC6\nEmission Reductions Achieved by 2050, Extended to 2300',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive plot saved: {output_file}")
    print()


def save_detailed_results(extracted, reduction_levels, output_file='magicc_results_detailed.csv'):
    """
    Save detailed results to CSV
    
    Parameters:
    -----------
    extracted : dict
        Extracted climate variables
    reduction_levels : list
        List of reduction levels
    output_file : str
        Output CSV filename
    """
    
    # Get all unique years from temperature data
    all_years = set()
    for level in reduction_levels:
        all_years.update(extracted['temperature'][level].index)
    
    if 'baseline' in extracted['temperature']:
        all_years.update(extracted['temperature']['baseline'].index)
    
    years = sorted(list(all_years))
    
    # Create DataFrame
    data = {'Year': years}
    
    # Add baseline
    if 'baseline' in extracted['temperature']:
        data['Baseline_Temp_C'] = [
            extracted['temperature']['baseline'].loc[y] if y in extracted['temperature']['baseline'].index else np.nan
            for y in years
        ]
        data['Baseline_CO2_ppm'] = [
            extracted['co2_concentration']['baseline'].loc[y] if y in extracted['co2_concentration']['baseline'].index else np.nan
            for y in years
        ]
    
    # Add each scenario
    for level in reduction_levels:
        pct = int(level * 100)
        
        data[f'Scenario_{pct}pct_Temp_C'] = [
            extracted['temperature'][level].loc[y] if y in extracted['temperature'][level].index else np.nan
            for y in years
        ]
        
        data[f'Scenario_{pct}pct_CO2_ppm'] = [
            extracted['co2_concentration'][level].loc[y] if y in extracted['co2_concentration'][level].index else np.nan
            for y in years
        ]
        
        data[f'Scenario_{pct}pct_Forcing_Wm2'] = [
            extracted['forcing'][level].loc[y] if y in extracted['forcing'][level].index else np.nan
            for y in years
        ]
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"✓ Detailed results saved: {output_file}")
    print()


def main():
    """
    Main execution function
    """
    print("=" * 80)
    print("ADVANCED CLIMATE MODEL USING PYMAGICC")
    print("=" * 80)
    print()
    print("This script uses MAGICC6, the same climate model used by the IPCC.")
    print("MAGICC provides sophisticated modeling of greenhouse gas cycles,")
    print("climate feedbacks, and radiative forcing.")
    print()
    print("=" * 80)
    print()
    
    # Initialize model
    model = AdvancedClimateModel()
    
    # Configuration
    reduction_levels = [0.25, 0.50, 0.75, 1.00]
    trough_year = 2050
    
    print(f"Configuration:")
    print(f"  Emission reduction targets: {[int(r*100) for r in reduction_levels]}% of 2025 baseline")
    print(f"  Target achievement year: {trough_year}")
    print(f"  Simulation period: 2025-2300")
    print()
    print("=" * 80)
    print()
    
    # Create scenarios
    print("Creating emission scenarios...")
    scenarios, baseline_2025 = model.create_emission_scenarios(
        reduction_levels=reduction_levels,
        trough_year=trough_year
    )
    
    # Run MAGICC simulations
    results = model.run_scenarios(scenarios)
    
    # Also run baseline RCP4.5 scenario
    print("Running baseline scenario (RCP4.5)...")
    with MAGICC6() as magicc:
        baseline_result = magicc.run(rcp45)
    print("✓ Baseline simulation complete!")
    print()
    
    # Extract results
    print("Extracting results...")
    extracted = model.extract_results(results, baseline_result)
    print("✓ Results extracted!")
    print()
    
    # Print summary
    model.print_summary(extracted, reduction_levels)
    
    # Create visualizations
    print("=" * 80)
    print("Creating visualizations...")
    create_comprehensive_plots(extracted, reduction_levels, 
                               output_file='magicc_climate_scenarios.png')
    
    # Save detailed results
    print("Saving detailed results...")
    save_detailed_results(extracted, reduction_levels, 
                         output_file='magicc_results_detailed.csv')
    
    print("=" * 80)
    print("ALL SIMULATIONS COMPLETE!")
    print("=" * 80)
    print()
    print("Output files created:")
    print("  1. magicc_climate_scenarios.png - Comprehensive visualization")
    print("  2. magicc_results_detailed.csv - Detailed numerical results")
    print()
    print("Thank you for using the Advanced Climate Model!")
    print("=" * 80)


if __name__ == "__main__":
    main()
