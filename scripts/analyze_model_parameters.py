"""
Analyze parameter composition of pHNN_Canonical model

Provides detailed breakdown of trainable parameters.
"""

import torch
import sys
sys.path.append('src')

from pHNN_canonical import pHNN_Canonical


def count_parameters(model, name="Model"):
    """Count and display parameters by component."""
    print(f"\n{'='*70}")
    print(f"{name} Parameter Breakdown")
    print(f"{'='*70}")

    total_params = 0
    component_params = {}

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        # Group by component
        if 'M_net' in name:
            component = 'Mass Matrix M(q)'
        elif 'H_net' in name:
            component = 'Hamiltonian H(q,p)'
        elif 'R_diag_raw' in name:
            component = 'Dissipation R'
        else:
            component = 'Other'

        if component not in component_params:
            component_params[component] = []

        component_params[component].append((name, param.shape, num_params))

    # Print by component
    for component in sorted(component_params.keys()):
        params_list = component_params[component]
        component_total = sum(p[2] for p in params_list)

        print(f"\n{component}: {component_total:,} parameters ({100*component_total/total_params:.1f}%)")
        print("-" * 70)

        for param_name, shape, num_params in params_list:
            # Simplify name
            simple_name = param_name.replace('M_net.', '').replace('H_net.', '').replace('_net.', '')
            print(f"  {simple_name:40s} {str(shape):20s} {num_params:6,}")

    print(f"\n{'='*70}")
    print(f"Total Parameters: {total_params:,}")
    print(f"{'='*70}\n")

    return total_params


def analyze_mass_matrix(model):
    """Analyze mass matrix structure."""
    print(f"\n{'='*70}")
    print(f"Mass Matrix Details")
    print(f"{'='*70}")

    M_net = model.M_net
    print(f"Type: {type(M_net).__name__}")

    if hasattr(M_net, 'log_a'):
        # CartPoleMassMatrix
        print("\nStructure: M(θ) = [[a, b*cos(θ)], [b*cos(θ), c]]")
        print("\nLearnable parameters:")
        print(f"  log_a: 1 parameter (a = exp(log_a))")
        print(f"  b:     1 parameter (coupling)")
        print(f"  log_c: 1 parameter (c = exp(log_c))")
        print(f"  Total: 3 parameters")

        # Show current values
        with torch.no_grad():
            a = torch.exp(M_net.log_a) + 1e-3
            b = M_net.b
            c = torch.exp(M_net.log_c) + 1e-3

            print(f"\nCurrent values:")
            print(f"  a = {a.item():.6f}")
            print(f"  b = {b.item():.6f}")
            print(f"  c = {c.item():.6f}")


def analyze_hamiltonian(model):
    """Analyze Hamiltonian network structure."""
    print(f"\n{'='*70}")
    print(f"Hamiltonian Network Details")
    print(f"{'='*70}")

    H_net = model.H_net

    print("\nArchitecture: MLP")
    print(f"Input:  4 (canonical coordinates [q, p])")
    print(f"Hidden: [128, 128]")
    print(f"Output: 1 (scalar energy)")

    print("\nLayer breakdown:")
    total = 0
    for name, param in H_net.named_parameters():
        shape = param.shape
        num = param.numel()
        total += num
        print(f"  {name:30s} {str(shape):20s} {num:6,}")

    print(f"\nTotal Hamiltonian parameters: {total:,}")


def analyze_r_matrix(model):
    """Analyze R matrix structure."""
    print(f"\n{'='*70}")
    print(f"Dissipation Matrix R Details")
    print(f"{'='*70}")

    print("\nStructure: Diagonal matrix")
    print(f"R = diag(r1, r2, r3, r4)")
    print(f"\nLearnable parameters: 4 (diagonal elements)")

    with torch.no_grad():
        R_diag = torch.nn.functional.softplus(model.R_diag_raw) + 1e-4
        print(f"\nCurrent diagonal values:")
        for i, val in enumerate(R_diag):
            print(f"  r{i+1} = {val.item():.6f}")


def main():
    print("\n" + "="*70)
    print("pHNN_Canonical Parameter Analysis")
    print("="*70)

    # Load model
    config_path = "cartpole_mpc_config.yaml"
    print(f"\nLoading model from config: {config_path}")

    model = pHNN_Canonical(config_path)

    # Overall parameter count
    total_params = count_parameters(model, "pHNN_Canonical")

    # Detailed analysis
    analyze_mass_matrix(model)
    analyze_hamiltonian(model)
    analyze_r_matrix(model)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"\nStructural parameters (PHS components):")
    print(f"  Mass matrix M(q):        3 parameters")
    print(f"  Dissipation R:           4 parameters")
    print(f"  Total structural:        7 parameters (0.04% of total)")

    print(f"\nLearnable energy function:")
    print(f"  Hamiltonian H(q,p):  {total_params - 7:,} parameters (99.96% of total)")

    print(f"\nFixed (non-learnable):")
    print(f"  J matrix:                Canonical (fixed)")
    print(f"  G matrix:                Fixed [0, 0, 1, 0]")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
