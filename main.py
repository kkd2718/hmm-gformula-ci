#!/usr/bin/env python
"""
main.py - Simulation Study Entry Point (v3.3)

Korean-Calibrated Hidden Markovian g-formula Simulation (20-Year)

Experiments:
    1. Effect Size Sensitivity (Parameter Recovery)
    2. Sample Size Robustness (5K ~ 100K)
    3. Model Selection & Ablation Study (LRT, AIC, BIC)
    4. 3-Way Method Comparison (Cox PH vs Markov vs HMM)
    5. Causal Inference with Bootstrap CI

Usage:
    python main.py --validate           # Validate config parameters
    python main.py --exp 1              # Run experiment 1
    python main.py --exp 3              # Run model selection (LRT)
    python main.py --exp 4              # Run 3-way comparison (Cox vs Markov vs HMM)
    python main.py --exp 5 --n-boot 200 # Full bootstrap CI
    python main.py --advanced           # Run spline curve analysis
    python main.py --all                # Run all experiments (1-5)
    python main.py --quick              # Quick test mode
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description='Korean-Calibrated HMM g-formula Simulation Study v3.3 (20-Year)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --validate              # Check parameter calibration
    python main.py --quick                 # Quick test (small N, few iterations)
    python main.py --exp 1                 # Parameter recovery by effect size
    python main.py --exp 2                 # Sample size robustness
    python main.py --exp 3                 # Model selection (LRT, AIC, BIC)
    python main.py --exp 4                 # 3-way comparison (Cox vs Markov vs HMM)
    python main.py --exp 5 --n-boot 200    # Causal inference with bootstrap
    python main.py --advanced              # Spline curve analysis with 95% CI
    python main.py --all                   # Run all experiments sequentially
        """
    )
    
    parser.add_argument('--validate', action='store_true', 
                        help='Validate configuration parameters')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3, 4, 5], 
                        help='Run specific experiment (1-5)')
    parser.add_argument('--advanced', action='store_true', 
                        help='Run advanced spline analysis with 95% CI')
    parser.add_argument('--all', action='store_true', 
                        help='Run all experiments (1-5)')
    parser.add_argument('--n-samples', type=int, default=20000, 
                        help='Sample size (default: 20000)')
    parser.add_argument('--n-sim', type=int, default=200, 
                        help='MC simulations (default: 200)')
    parser.add_argument('--n-boot', type=int, default=200, 
                        help='Bootstrap iterations (default: 200)')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick test mode (reduced iterations)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Hidden Markovian g-formula Simulation Study v3.3")
    print("Korean Population Calibrated (KNHANES 2023, KDCA 2022)")
    print("20-Year Longitudinal Simulation")
    print("="*70)
    
    # Validate config
    if args.validate:
        from config import validate_config
        validate_config()
        return
    
    # Quick mode overrides
    if args.quick:
        args.n_samples = 5000
        args.n_sim = 30
        args.n_boot = 50
        print("\n[Quick Test Mode]")
        print(f"  N samples: {args.n_samples}")
        print(f"  MC simulations: {args.n_sim}")
        print(f"  Bootstrap: {args.n_boot}")
    
    # Run advanced analysis (Spline curves)
    if args.advanced:
        from analysis_advanced_prs import run_advanced_analysis
        run_advanced_analysis(
            n_samples=args.n_samples,
            n_bootstrap=args.n_boot,
            save_results=True,
        )
        return
    
    # Run all experiments
    if args.all:
        from experiments.run_experiments import run_all_experiments
        run_all_experiments(
            n_simulations=args.n_sim,
            n_samples=args.n_samples,
            quick=args.quick,
        )
        return
    
    # Run specific experiment
    if args.exp == 1:
        from experiments.run_experiments import run_experiment_1
        print("\n[Experiment 1: Effect Size Sensitivity]")
        run_experiment_1(
            n_simulations=args.n_sim,
            n_samples=args.n_samples,
        )
        
    elif args.exp == 2:
        from experiments.run_experiments import run_experiment_2
        print("\n[Experiment 2: Sample Size Robustness]")
        print("Sample sizes: 5,000 / 10,000 / 20,000 / 50,000 / 100,000")
        run_experiment_2(
            n_simulations=args.n_sim,
        )
        
    elif args.exp == 3:
        from experiments.run_experiments import run_experiment_3
        print("\n[Experiment 3: Model Selection & Ablation Study]")
        print("Comparing: Full Model vs No Interaction vs No Pack-years")
        print("Metrics: Log-Likelihood, AIC, BIC, LRT P-value")
        run_experiment_3(
            n_simulations=args.n_sim,
            n_samples=args.n_samples,
        )
        
    elif args.exp == 4:
        from experiments.run_experiments import run_experiment_4
        print("\n[Experiment 4: 3-Way Method Comparison]")
        print("Comparing:")
        print("  A. Time-varying Cox PH (Hazard Ratio)")
        print("  B. Standard Markov g-formula (Risk Ratio, no latent Z)")
        print("  C. Proposed HMM g-formula (Risk Ratio, with latent Z)")
        print("\nGoal: Demonstrate that Cox/Markov underestimate GxE")
        print("      while HMM corrects for unmeasured confounding (Sick-quitter bias)")
        run_experiment_4(
            n_simulations=args.n_sim,
            n_samples=args.n_samples,
        )
        
    elif args.exp == 5:
        from experiments.run_experiments import run_experiment_5
        print("\n[Experiment 5: Causal Inference with Bootstrap CI]")
        run_experiment_5(
            n_samples=args.n_samples,
            n_bootstrap=args.n_boot,
        )
        
    else:
        # Default: Run Experiment 5
        from experiments.run_experiments import run_experiment_5
        print("\n[Default: Running Experiment 5 - Causal Inference]")
        run_experiment_5(
            n_samples=args.n_samples,
            n_bootstrap=args.n_boot,
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()