"""
Simple Climate Scenario Simulator
===================================
Simulates CO2 emissions plateaus at 25%, 50%, 75%, and 100% of current emissions
out to the year 2300 using a simplified climate model.

This script does NOT require pymagicc or Wine - it uses a simple energy balance
climate model with carbon cycle to simulate temperature and CO2 concentration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class SimpleClimateModel:
    """
    Simplified climate model based on energy balance and carbon cycle
    """

    def __init__(self):
        # Climate sensitivity parameters
        self.ECS = 3.0  # Equilibrium Climate Sensitivity (°C for doubling CO2)
        self.lambda_param = 3.7 / self.ECS  # Climate feedback parameter (W/m²/K)

        # Carbon cycle parameters (simplified box model)
        self.co2_pre_industrial = 280.0  # ppm
        self.co2_current = 420.0  # ppm (2024)

        # Airborne fraction and decay
        self.airborne_fraction = 0.45  # Fraction of emissions staying in atmosphere
        self.decay_rate = 0.02  # Annual decay rate for excess CO2

        # Ocean heat uptake
        self.ocean_heat_capacity = 100.0  # Years of heat capacity

    def emissions_to_concentration(self, emissions_gtc, current_co2, dt=1.0):
        """
        Convert emissions (GtC/yr) to change in CO2 concentration (ppm)

        Parameters:
        -----------
        emissions_gtc : float
            Emissions in GtC/yr
        current_co2 : float
            Current CO2 concentration in ppm
        dt : float
            Time step in years

        Returns:
        --------
        delta_co2 : float
            Change in CO2 concentration (ppm)
        """
        # Convert GtC to ppm (1 GtC ~ 0.128 ppm per year in atmosphere)
        # Accounting for airborne fraction
        emissions_ppm = emissions_gtc * 0.128 * self.airborne_fraction

        # Natural decay of excess CO2
        excess_co2 = current_co2 - self.co2_pre_industrial
        decay_ppm = excess_co2 * self.decay_rate

        delta_co2 = (emissions_ppm - decay_ppm) * dt

        return delta_co2

    def co2_to_forcing(self, co2_concentration):
        """
        Convert CO2 concentration to radiative forcing

        Parameters:
        -----------
        co2_concentration : float
            CO2 concentration in ppm

        Returns:
        --------
        forcing : float
            Radiative forcing in W/m²
        """
        # Logarithmic relationship
        forcing = 5.35 * np.log(co2_concentration / self.co2_pre_industrial)
        return forcing

    def forcing_to_temperature(self, forcing, current_temp, dt=1.0):
        """
        Convert radiative forcing to temperature change

        Parameters:
        -----------
        forcing : float
            Radiative forcing in W/m²
        current_temp : float
            Current temperature anomaly in °C
        dt : float
            Time step in years

        Returns:
        --------
        delta_temp : float
            Change in temperature (°C)
        """
        # Energy balance: dT/dt = (F - λT) / C
        # where F is forcing, λ is feedback parameter, C is heat capacity
        equilibrium_temp = forcing / self.lambda_param
        delta_temp = (equilibrium_temp - current_temp) / self.ocean_heat_capacity * dt

        return delta_temp

    def run_scenario(self, emissions_profile, years):
        """
        Run climate simulation for a given emissions profile

        Parameters:
        -----------
        emissions_profile : array
            Emissions in GtC/yr for each year
        years : array
            Years corresponding to emissions

        Returns:
        --------
        results : dict
            Dictionary containing time series of CO2, temperature, and forcing
        """
        n_years = len(years)

        # Initialize arrays
        co2 = np.zeros(n_years)
        temp = np.zeros(n_years)
        forcing = np.zeros(n_years)

        # Initial conditions (year 2024)
        co2[0] = self.co2_current
        temp[0] = 1.2  # Current warming ~1.2°C above pre-industrial
        forcing[0] = self.co2_to_forcing(co2[0])

        # Time stepping
        for i in range(1, n_years):
            dt = years[i] - years[i-1]

            # Update CO2 concentration
            delta_co2 = self.emissions_to_concentration(emissions_profile[i-1], co2[i-1], dt)
            co2[i] = max(co2[i-1] + delta_co2, self.co2_pre_industrial)

            # Update forcing
            forcing[i] = self.co2_to_forcing(co2[i])

            # Update temperature
            delta_temp = self.forcing_to_temperature(forcing[i], temp[i-1], dt)
            temp[i] = temp[i-1] + delta_temp

        results = {
            'years': years,
            'co2': co2,
            'temperature': temp,
            'forcing': forcing,
            'emissions': emissions_profile
        }

        return results


def create_emission_scenario(baseline_gtc, reduction_fraction, years,
                            start_year=2025, plateau_year=2050):
    """
    Create an emissions scenario with linear transition to plateau

    Parameters:
    -----------
    baseline_gtc : float
        Baseline emissions in GtC/yr
    reduction_fraction : float
        Fraction of baseline to maintain (e.g., 0.25 = 25% of baseline)
    years : array
        Array of years
    start_year : int
        Year to start reduction
    plateau_year : int
        Year to reach plateau level

    Returns:
    --------
    emissions : array
        Emissions profile in GtC/yr
    """
    emissions = np.zeros(len(years))
    target_emission = baseline_gtc * reduction_fraction

    for i, year in enumerate(years):
        if year < start_year:
            # Before start: use baseline
            emissions[i] = baseline_gtc
        elif year < plateau_year:
            # Linear transition to target
            progress = (year - start_year) / (plateau_year - start_year)
            emissions[i] = baseline_gtc - (baseline_gtc - target_emission) * progress
        else:
            # At or after plateau: maintain target
            emissions[i] = target_emission

    return emissions


def main():
    print("=" * 80)
    print("SIMPLE CLIMATE SCENARIO SIMULATOR")
    print("=" * 80)
    print()
    print("Simulating CO2 emissions plateaus at 25%, 50%, 75%, and 100%")
    print("of current emissions out to year 2300.")
    print()
    print("Using simplified energy balance climate model with carbon cycle.")
    print("=" * 80)
    print()

    # Initialize model
    model = SimpleClimateModel()

    # Configuration
    baseline_emissions = 10.7  # GtC/yr (approximately 39 GtCO2/yr)
    reduction_levels = [0.25, 0.50, 0.75, 1.00]
    start_year = 2025
    plateau_year = 2050

    # Time array
    years = np.arange(2024, 2301)

    print(f"Configuration:")
    print(f"  Baseline emissions: {baseline_emissions:.1f} GtC/yr (~{baseline_emissions*3.67:.0f} GtCO2/yr)")
    print(f"  Plateau levels: {[int(r*100) for r in reduction_levels]}% of baseline")
    print(f"  Start year: {start_year}")
    print(f"  Plateau year: {plateau_year}")
    print(f"  Simulation period: {years[0]}-{years[-1]}")
    print()
    print("=" * 80)
    print()

    # Run scenarios
    results = {}
    print("Running simulations...")

    for i, level in enumerate(reduction_levels):
        print(f"  [{i+1}/{len(reduction_levels)}] {int(level*100)}% scenario...")

        # Create emissions profile
        emissions = create_emission_scenario(baseline_emissions, level, years,
                                            start_year, plateau_year)

        # Run model
        result = model.run_scenario(emissions, years)
        results[level] = result

    print("✓ All simulations complete!")
    print()

    # Print summary
    print("=" * 80)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 80)
    print()

    for level in reduction_levels:
        result = results[level]
        temp = result['temperature']
        co2 = result['co2']
        emissions = result['emissions']
        years_array = result['years']

        print(f"{'─' * 80}")
        print(f"SCENARIO: {int(level*100)}% PLATEAU ({int((1-level)*100)}% reduction)")
        print(f"{'─' * 80}")
        print()

        # Find indices for key years
        idx_2050 = np.where(years_array == 2050)[0][0]
        idx_2100 = np.where(years_array == 2100)[0][0]
        idx_2200 = np.where(years_array == 2200)[0][0]
        idx_2300 = np.where(years_array == 2300)[0][0]

        print(f"Temperature (°C above pre-industrial):")
        print(f"  2050: {temp[idx_2050]:.2f}°C")
        print(f"  2100: {temp[idx_2100]:.2f}°C")
        print(f"  2200: {temp[idx_2200]:.2f}°C")
        print(f"  2300: {temp[idx_2300]:.2f}°C")
        print(f"  Peak: {temp.max():.2f}°C (Year {years_array[temp.argmax()]})")
        print()

        print(f"CO2 Concentration (ppm):")
        print(f"  2050: {co2[idx_2050]:.1f} ppm")
        print(f"  2100: {co2[idx_2100]:.1f} ppm")
        print(f"  2200: {co2[idx_2200]:.1f} ppm")
        print(f"  2300: {co2[idx_2300]:.1f} ppm")
        print(f"  Peak: {co2.max():.1f} ppm (Year {years_array[co2.argmax()]})")
        print()

        # Paris Agreement assessment
        temp_2100 = temp[idx_2100]
        if temp_2100 < 1.5:
            status = "✓ Stays below 1.5°C Paris target"
        elif temp_2100 < 2.0:
            status = "⚠ Exceeds 1.5°C but stays below 2.0°C Paris limit"
        else:
            status = "✗ Exceeds 2.0°C Paris limit"
        print(f"Paris Agreement Status (2100): {status}")
        print()

    # Create comprehensive plots
    print("=" * 80)
    print("Creating visualizations...")
    print()

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Color scheme
    colors = {
        0.25: '#2E86AB',
        0.50: '#A23B72',
        0.75: '#F18F01',
        1.00: '#C73E1D'
    }

    labels = {
        0.25: '25% plateau (75% reduction)',
        0.50: '50% plateau (50% reduction)',
        0.75: '75% plateau (25% reduction)',
        1.00: '100% plateau (no reduction)'
    }

    # Plot 1: Temperature trajectories
    ax1 = fig.add_subplot(gs[0, :])
    for level in reduction_levels:
        result = results[level]
        ax1.plot(result['years'], result['temperature'], '-', linewidth=2.5,
                color=colors[level], label=labels[level])

    ax1.axhline(y=1.5, color='orange', linestyle='--', alpha=0.6, linewidth=2, label='1.5°C Paris Target')
    ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.6, linewidth=2, label='2.0°C Paris Limit')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Temperature Anomaly (°C above pre-industrial)', fontsize=12)
    ax1.set_title('Global Temperature Projections - Simplified Climate Model', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2020, 2300)

    # Plot 2: CO2 Concentration
    ax2 = fig.add_subplot(gs[1, 0])
    for level in reduction_levels:
        result = results[level]
        ax2.plot(result['years'], result['co2'], '-', linewidth=2,
                color=colors[level], label=labels[level])

    ax2.axhline(y=280, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='Pre-industrial')
    ax2.axhline(y=420, color='orange', linestyle=':', alpha=0.5, linewidth=1, label='Current (~2024)')
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Atmospheric CO₂ (ppm)', fontsize=11)
    ax2.set_title('CO₂ Concentration', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2020, 2300)

    # Plot 3: Radiative Forcing
    ax3 = fig.add_subplot(gs[1, 1])
    for level in reduction_levels:
        result = results[level]
        ax3.plot(result['years'], result['forcing'], '-', linewidth=2,
                color=colors[level], label=labels[level])

    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('Radiative Forcing (W/m²)', fontsize=11)
    ax3.set_title('CO₂ Radiative Forcing', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(2020, 2300)

    # Plot 4: Temperature in 2100
    ax4 = fig.add_subplot(gs[2, 0])
    temps_2100 = []
    scenario_labels = []
    bar_colors = []

    for level in reduction_levels:
        result = results[level]
        idx_2100 = np.where(result['years'] == 2100)[0][0]
        temps_2100.append(result['temperature'][idx_2100])
        scenario_labels.append(f"{int(level*100)}%")
        bar_colors.append(colors[level])

    bars = ax4.bar(range(len(temps_2100)), temps_2100, color=bar_colors, alpha=0.8, edgecolor='black')
    ax4.axhline(y=1.5, color='orange', linestyle='--', alpha=0.6, linewidth=2, label='1.5°C Target')
    ax4.axhline(y=2.0, color='red', linestyle='--', alpha=0.6, linewidth=2, label='2.0°C Limit')
    ax4.set_xticks(range(len(scenario_labels)))
    ax4.set_xticklabels(scenario_labels)
    ax4.set_xlabel('Emission Plateau Level', fontsize=11)
    ax4.set_ylabel('Temperature (°C)', fontsize=11)
    ax4.set_title('Temperature in 2100 by Scenario', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, temps_2100)):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}°C',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 5: Emissions trajectories
    ax5 = fig.add_subplot(gs[2, 1])
    for level in reduction_levels:
        result = results[level]
        ax5.plot(result['years'], result['emissions'], '-', linewidth=2,
                color=colors[level], label=labels[level])

    ax5.set_xlabel('Year', fontsize=11)
    ax5.set_ylabel('CO₂ Emissions (GtC/yr)', fontsize=11)
    ax5.set_title('Emission Scenarios', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(2020, 2150)  # Focus on transition period

    # Add overall title
    fig.suptitle('Climate Scenario Analysis: CO₂ Emission Plateaus to 2300',
                fontsize=16, fontweight='bold', y=0.995)

    output_file = 'climate_scenarios_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_file}")
    print()

    # Save detailed results to CSV
    print("Saving detailed results to CSV...")

    # Create combined dataframe
    data = {'Year': years}
    for level in reduction_levels:
        result = results[level]
        pct = int(level * 100)
        data[f'{pct}%_Emissions_GtC'] = result['emissions']
        data[f'{pct}%_CO2_ppm'] = result['co2']
        data[f'{pct}%_Temp_C'] = result['temperature']
        data[f'{pct}%_Forcing_Wm2'] = result['forcing']

    df = pd.DataFrame(data)
    csv_file = 'climate_scenarios_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"✓ Detailed results saved: {csv_file}")
    print()

    print("=" * 80)
    print("ALL SIMULATIONS COMPLETE!")
    print("=" * 80)
    print()
    print("Output files created:")
    print(f"  1. {output_file} - Comprehensive visualization")
    print(f"  2. {csv_file} - Detailed numerical results")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
