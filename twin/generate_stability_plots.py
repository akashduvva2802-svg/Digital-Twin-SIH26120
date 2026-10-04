import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

def analyze_mlp_spectral_stability(model: nn.Module, model_name: str, out_dir: str):
    layer_indices = []
    max_eigenvalues = []
    lipschitz_bound = 1.0
    layer_idx = 1
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.detach()
            S = torch.linalg.svdvals(W)
            max_eig_val = S.max().item()
            
            layer_indices.append(layer_idx)
            max_eigenvalues.append(max_eig_val)
            lipschitz_bound *= max_eig_val
            layer_idx += 1
            
    plt.figure(figsize=(8, 5))
    plt.bar(layer_indices, max_eigenvalues, color='skyblue', edgecolor='black')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Neutral Threshold (1.0)')
    plt.xlabel('Layer Index')
    plt.ylabel('Maximum Eigenvalue (Spectral Norm)')
    plt.title(f'{model_name}: Spectral Norm per Layer\nGlobal Lipschitz Bound: {lipschitz_bound:.4f}')
    plt.xticks(layer_indices)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{model_name}_spectral_stability.png")
    plt.savefig(plot_path)
    plt.close()
    
    return pd.DataFrame({
        'Model': model_name,
        'Layer': layer_indices,
        'Max_Eigenvalue': max_eigenvalues
    }), lipschitz_bound


def analyze_jacobian_eigenvalues(model: nn.Module, dummy_input: torch.Tensor, model_name: str, out_dir: str):
    dummy_input.requires_grad_(True)
    output = model(dummy_input)
    
    if isinstance(output, tuple) or isinstance(output, dict):
        out_tensor = list(output.values())[0] if isinstance(output, dict) else output[0]
    else:
        out_tensor = output

    jacobian_max_eigs = []
    
    for i in range(out_tensor.shape[0]):
        grad_out = torch.zeros_like(out_tensor)
        grad_out[i] = 1.0
        
        J_row = torch.autograd.grad(outputs=out_tensor, inputs=dummy_input, 
                                    grad_outputs=grad_out, retain_graph=True, create_graph=True)[0]
        
        J_i = J_row[i].unsqueeze(0)
        JT_J = torch.matmul(J_i.T, J_i)
        eigenvalues = torch.linalg.eigvals(JT_J).real
        max_eig = eigenvalues.max().item()
        jacobian_max_eigs.append(max_eig)
        
    avg_max_eig = np.mean(jacobian_max_eigs)
    
    plt.figure(figsize=(8, 5))
    plt.hist(jacobian_max_eigs, bins=20, color='coral', edgecolor='black')
    plt.axvline(x=avg_max_eig, color='k', linestyle='dashed', linewidth=1.5, label=f'Mean Max Eig: {avg_max_eig:.4f}')
    plt.xlabel('Maximum Local Eigenvalue (Sensitivity)')
    plt.ylabel('Frequency (Test Points)')
    plt.title(f'{model_name}: Jacobian Eigenvalue Distribution\n(Input Sensitivity / Gradient Bias)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{model_name}_jacobian_stability.png")
    plt.savefig(plot_path)
    plt.close()
    
    return avg_max_eig, jacobian_max_eigs


if __name__ == "__main__":
    out_dir = "C:\\Akash\\PINN_AND_DATA\\scada_pinn\\baghewala_scada_twin\\scada\\twin\\analysis_out"
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        from pinn_digital_twin import ReservoirPINN, SRPPINN
        print("Loading models and real data...")
        
        # Load Reservoir PINN
        res_model = ReservoirPINN()
        
        # Load SRP PINN
        df_srp_data = pd.read_csv('srp_dynamometer_microscale.csv')
        srp_model = SRPPINN(df_srp_data)
        
        # Load pre-trained weights to analyze the ACTUAL model!
        print("Loading pretrained weights...")
        ckpt = torch.load('digital_twin_models.pt', map_location='cpu')
        res_model.load_state_dict(ckpt['reservoir'])
        srp_model.load_state_dict(ckpt['srp'], strict=False)
        
        print("Running Spectral Analysis...")
        df_res, L_res = analyze_mlp_spectral_stability(res_model, "ReservoirPINN", out_dir)
        df_srp, L_srp = analyze_mlp_spectral_stability(srp_model, "SRPPINN", out_dir)
        
        df_res.to_csv(os.path.join(out_dir, "ReservoirPINN_Spectral.csv"), index=False)
        df_srp.to_csv(os.path.join(out_dir, "SRPPINN_Spectral.csv"), index=False)
        
        print("Running Jacobian Analysis...")
        dummy_in_res = torch.rand(100, 1) * 300.0 # tau from 0 to 300 days
        dummy_in_res_params = {
            "rh": torch.ones(100, 1) * 20.0, "h": torch.ones(100, 1) * 18.0, 
            "k_md": torch.ones(100, 1) * 130.0, "mu_cold": torch.ones(100, 1) * 13.0, 
            "mu_hot": torch.ones(100, 1) * 0.35, "T0": torch.ones(100, 1) * 320.0, 
            "Ts": torch.ones(100, 1) * 523.0, "alpha": torch.ones(100, 1) * 3.7e-7
        }
        
        class ResWrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, tau): return self.m(tau, dummy_in_res_params)["T"]
            
        res_wrapper = ResWrapper(res_model)
        avg_eig_res, _ = analyze_jacobian_eigenvalues(res_wrapper, dummy_in_res, "ReservoirPINN", out_dir)
        
        # SRPPINN Wrapper
        # SRPPINN signature: forward(self, t, xi, pw, tau=1.0)
        dummy_in_srp = torch.rand(100, 1) * 4.0 # Time within stroke
        class SRPWrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, t): 
                return self.m.net(torch.cat([t, torch.zeros(100, 2)], dim=1))[:, 0]
                
        srp_wrapper = SRPWrapper(srp_model)
        avg_eig_srp, _ = analyze_jacobian_eigenvalues(srp_wrapper, dummy_in_srp, "SRPPINN", out_dir)
        
        # Print results to stdout to be captured by the AI agent
        print("\n--- STABILITY RESULTS ---")
        print(f"ReservoirPINN Lipschitz Bound: {L_res:.4f}")
        print(f"SRPPINN Lipschitz Bound: {L_srp:.4f}")
        print(f"ReservoirPINN Avg Jacobian Max Eig: {avg_eig_res:.4f}")
        print(f"SRPPINN Avg Jacobian Max Eig: {avg_eig_srp:.4f}")
        
        # Check learned damping (c parameter)
        try:
            c = srp_model.c.item()
            print(f"SRPPINN Learned Damping (c): {c:.4f}")
            # Characteristic equation roots: lambda^2 + c*lambda + k = 0
            # k = (1.0)^2 = 1.0 (normalized wave speed in the dimensionless PDE)
            k = 1.0
            discriminant = c**2 - 4*k
            if discriminant >= 0:
                l1 = (-c + np.sqrt(discriminant))/2
                l2 = (-c - np.sqrt(discriminant))/2
                print(f"PDE Eigenvalues (Real): {l1:.4f}, {l2:.4f}")
            else:
                real_part = -c/2
                imag_part = np.sqrt(abs(discriminant))/2
                print(f"PDE Eigenvalues (Complex): {real_part:.4f} +/- {imag_part:.4f}i")
        except AttributeError:
            pass

        print("DONE")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
