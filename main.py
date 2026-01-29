#!/usr/bin/env python
"""
main.py - Simulation Study Entry Point (v3.1)

Korean-Calibrated Hidden Markovian g-formula Simulation

Settings:
- Sample sizes: 5K ~ 100K (for rare CVD outcome ~3-4%)
- MC simulations: 200 per scenario (consistent across all N)
- Bootstrap CI: 200 iterations

Usage:
    python main.py                        # Run main analysis (Exp 5)
    python main.py --validate             # Validate config parameters
    python main.py --exp 1                # Run parameter recovery
    python main.py --exp 2                # Run sample size robustness
    python main.py --exp 5 --n-boot 500   # Full bootstrap CI
    python main.py --advanced             # Run spline curve analysis
    python main.py --all                  # Run all experiments
    python main.py --quick                # Quick test mode
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description='Korean-Calibrated HMM g-formula Simulation Study v3.1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --validate             # Check parameter calibration
    python main.py --quick                # Quick test (small N, few iterations)
    python main.py --exp 2                # Sample size robustness (5K-100K)
    python main.py --exp 5 --n-boot 200   # Causal inference with bootstrap
    python main.py --advanced             # Spline curve analysis
    python main.py --all                  # Run all experiments
        """
    )
    
    parser.add_argument('--validate', action='store_true', 
                        help='Validate configuration parameters')
    parser.add_argument('--exp', type=int, choices=[1, 2, 5], 
                        help='Run specific experiment')
    parser.add_argument('--advanced', action='store_true', 
                        help='Run advanced spline analysis')
    parser.add_argument('--all', action='store_true', 
                        help='Run all experiments')
    parser.add_argument('--n-samples', type=int, default=20000, 
                        help='Sample size (default: 20000)')
    parser.add_argument('--n-sim', type=int, default=200, 
                        help='MC simulations (default: 200)')
    parser.add_argument('--n-boot', type=int, default=200, 
                        help='Bootstrap iterations (default: 200)')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick test mode')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Hidden Markovian g-formula Simulation Study v3.1")
    print("Korean Population Calibrated (KNHANES 2023, KDCA 2022)")
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
    
    # Run advanced analysis
    if args.advanced:
        from analysis_advanced import run_advanced_analysis
        run_advanced_analysis(
            n_samples=args.n_samples,
            save_results=True,
        )
        return
    
    # Run all experiments
    if args.all:
        from experiments.run_experiments import run_all_experiments
        run_all_experiments(
            n_simulations=args.n_sim,
            quick=args.quick,
        )
        return
    
    # Run specific experiment
    if args.exp == 1:
        from experiments.run_experiments import run_experiment_1
        run_experiment_1(
            n_simulations=args.n_sim,
            n_samples=args.n_samples,
        )
    elif args.exp == 2:
        from experiments.run_experiments import run_experiment_2
        print("\n[Experiment 2: Sample Size Robustness]")
        print("Sample sizes: 5,000 / 10,000 / 20,000 / 50,000 / 100,000")
        print("(Designed for rare CVD outcome with ~3-4% 10-year incidence)")
        run_experiment_2(
            n_simulations=args.n_sim,
        )
    elif args.exp == 5:
        from experiments.run_experiments import run_experiment_5
        run_experiment_5(
            n_samples=args.n_samples,
            n_bootstrap=args.n_boot,
        )
    else:
        # Default: Run Experiment 5 (main causal analysis)
        from experiments.run_experiments import run_experiment_5
        print("\n[Default: Running Experiment 5 - Causal Inference]")
        run_experiment_5(
            n_samples=args.n_samples,
            n_bootstrap=args.n_boot,
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()