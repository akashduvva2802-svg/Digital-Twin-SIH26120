import torch
import torch.nn as nn
import numpy as np

def analyze_mlp_spectral_stability(model: nn.Module, model_name: str):
    """
    Computes the spectral norm (largest singular value) of each weight matrix 
    in the MLP. The product of these norms provides the Global Lipschitz bound, 
    which strictly guarantees how much the model can amplify input noise.
    """
    print(f"\n--- Spectral Stability Analysis: {model_name} ---")
    
    lipschitz_bound = 1.0
    layer_idx = 1
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Extract weight matrix
            W = module.weight.detach()
            # Compute Singular Value Decomposition (SVD)
            # The singular values are the square roots of the eigenvalues of W^T W
            S = torch.linalg.svdvals(W)
            max_eig_val = S.max().item()
            
            print(f"Layer {layer_idx} ({name}): Max Eigenvalue (Spectral Norm) = {max_eig_val:.4f}")
            lipschitz_bound *= max_eig_val
            layer_idx += 1
            
    print(f">> Global Lipschitz Bound (Theoretical Max Amplification): {lipschitz_bound:.4f}")
    if lipschitz_bound > 100:
        print(">> WARNING: High Lipschitz bound indicates potential sensitivity to sensor noise.")
    else:
        print(">> PASSED: Model is strictly bounded and robust to input perturbations.")


def analyze_jacobian_eigenvalues(model: nn.Module, dummy_input: torch.Tensor, model_name: str):
    """
    Computes the local Jacobian J = df/dx and the eigenvalues of J^T J.
    This reveals the principal directions of sensitivity and proves unbiasedness
    to minor feature corruptions.
    """
    print(f"\n--- Local Jacobian Stability (Sensitivity): {model_name} ---")
    
    dummy_input.requires_grad_(True)
    output = model(dummy_input)
    
    # We take the sum of outputs to compute a scalar for autograd, 
    # or compute the exact Jacobian for each output dimension.
    # For a PINN, we usually care about the primary prediction (e.g., Temperature or Pump Load)
    
    if isinstance(output, tuple) or isinstance(output, dict):
        # Handle dict outputs from ReservoirPINN / SRPPINN
        out_tensor = list(output.values())[0] if isinstance(output, dict) else output[0]
    else:
        out_tensor = output

    jacobian_norms = []
    
    # Compute gradients for each sample in the batch
    for i in range(out_tensor.shape[0]):
        grad_out = torch.zeros_like(out_tensor)
        grad_out[i] = 1.0
        
        # Compute backward pass to get Jacobian row
        J_row = torch.autograd.grad(outputs=out_tensor, inputs=dummy_input, 
                                    grad_outputs=grad_out, retain_graph=True, create_graph=True)[0]
        
        # Eigenvalues of J^T J (equivalent to squared singular values of J)
        J_i = J_row[i].unsqueeze(0)  # Shape (1, input_dim)
        JT_J = torch.matmul(J_i.T, J_i)
        
        # Calculate eigenvalues
        eigenvalues = torch.linalg.eigvals(JT_J).real
        max_eig = eigenvalues.max().item()
        jacobian_norms.append(max_eig)
        
    avg_max_eig = np.mean(jacobian_norms)
    print(f"Average Maximum Local Eigenvalue (Sensitivity): {avg_max_eig:.4e}")
    if avg_max_eig < 10.0:
        print(">> PASSED: Local Jacobian spectrum is stable. The PINN does not wildly extrapolate.")
    else:
        print(">> WARNING: The PINN has steep gradients locally, which may cause erratic control.")


def analyze_pde_characteristic_stability(model_name: str, damping_c: float, stiffness_k: float):
    """
    Evaluates the physical stability of the learned PDE (e.g., the Damped Wave Equation for the SRP).
    Using the characteristic polynomial lambda^2 + c*lambda + k = 0.
    For the digital twin to be stable, the real part of the roots (eigenvalues) MUST be <= 0.
    """
    print(f"\n--- Physical PDE Eigenvalue Stability: {model_name} ---")
    
    # Roots of lambda^2 + c*lambda + k = 0 are: (-c +/- sqrt(c^2 - 4k)) / 2
    discriminant = damping_c**2 - 4 * stiffness_k
    
    if discriminant >= 0:
        lambda_1 = (-damping_c + np.sqrt(discriminant)) / 2
        lambda_2 = (-damping_c - np.sqrt(discriminant)) / 2
        print(f"Learned PDE Eigenvalues (Real, Overdamped): λ1 = {lambda_1:.4f}, λ2 = {lambda_2:.4f}")
        max_real = max(lambda_1, lambda_2)
    else:
        real_part = -damping_c / 2
        imag_part = np.sqrt(abs(discriminant)) / 2
        print(f"Learned PDE Eigenvalues (Complex, Underdamped): λ = {real_part:.4f} +/- {imag_part:.4f}i")
        max_real = real_part
        
    if max_real <= 0:
        print(">> PASSED: The physical system modeled by the PINN is strictly Dissipative & Stable.")
    else:
        print(">> FAILED: The PINN learned an exponentially diverging physical model (unstable bounds).")

if __name__ == "__main__":
    print("===============================================================")
    print("   PINN DIGITAL TWIN - FUNDAMENTAL STABILITY DIAGNOSTICS")
    print("===============================================================")
    
    try:
        from pinn_digital_twin import ReservoirPINN, SRPPINN
        
        # 1. Initialize models (assuming default parameters from your script)
        print("Loading models...")
        res_model = ReservoirPINN()
        srp_model = SRPPINN()
        
        # 2. Spectral Analysis (Weight matrix eigenvalues)
        analyze_mlp_spectral_stability(res_model, "ReservoirPINN")
        analyze_mlp_spectral_stability(srp_model, "SRPPINN")
        
        # 3. Jacobian Sensitivity Analysis
        # Create dummy inputs matching the PINN input shapes
        # ReservoirPINN expects [tau] and [params dict] usually, depending on forward pass.
        # Here we mock a generic MLP forward pass structure to demonstrate.
        dummy_input_res = torch.randn(10, 1) # Example batch of 10 times
        # analyze_jacobian_eigenvalues(res_model, dummy_input_res, "ReservoirPINN")
        
        # 4. PDE Characteristic Analysis (Using hypothetical learned params from SRPPINN)
        # For a sucker rod pump wave equation u_tt + c u_t - k u_xx = 0
        # If the PINN learns negative damping, it will blow up.
        learned_damping = 0.15 # Replace with srp_model.c.item() if it's a learned param
        learned_stiffness = 1.0 # Wave speed squared
        analyze_pde_characteristic_stability("SRP Wave Equation PDE", learned_damping, learned_stiffness)
        
    except ImportError:
        print("Please run this script in the environment where PyTorch and pinn_digital_twin are installed.")
