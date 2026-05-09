# Physics-Informed Neural Networks for Hyperelasticity and Inverse Parameter Identification

This repository contains a Physics-Informed Neural Network (PINN) workflow for identifying hyperelastic material parameters from full-field displacement data. The code couples a neural network displacement surrogate with a finite-element-based residual evaluation so that material parameters can be learned from observed deformation fields.

## What this project does

- **Forward PINN**: predicts displacement fields while enforcing mechanics-based residuals.
- **Inverse PINN**: identifies bulk and shear moduli from displacement and force data.
- **Finite element support**: precomputes element shape-function information and residual terms.
- **Post-processing**: writes loss histories, residual histories, and saved model weights.

## Repository layout

```text
README.md
Source/
	Analysis_PINN_Model.py
	config.yaml
	enumerate_dof.py
	finite_element_model.py
	finite_element_model_fully_vectorized.py
	finite_element_model_fully_vectorised_inverse.py
	pinn_model.py
	pinn_model_inverse.py
	shape_functions.py
	Vtu_Data_Creation.py
	Model_With_Input_Class.py
	Load_nodesandmesh.m
	*.vtu
```

### Key scripts

- `Source/pinn_model.py`: trains the forward PINN displacement model.
- `Source/pinn_model_inverse.py`: trains the inverse PINN and learns material parameters.
- `Source/finite_element_model_fully_vectorized.py`: vectorized forward FEM/PINN residual computation.
- `Source/finite_element_model_fully_vectorised_inverse.py`: inverse residual and parameter loss computation.
- `Source/enumerate_dof.py`: maps mesh connectivity to degrees of freedom.
- `Source/config.yaml`: central configuration for paths, material values, and training settings.

## Data expectations

The scripts expect the following runtime data structure relative to the repository root:

```text
Input/
	Element512/
		nodes.txt
		connectivity.txt
		force_global_10_load_steps.txt
		displacement_0.050000.txt
		displacement_0.100000.txt
		...
Output/
	Element512/
```

If your dataset uses a different mesh name or different load-step files, update `Source/config.yaml` and the file names referenced in `Source/pinn_model.py`, `Source/pinn_model_inverse.py`, and `Source/enumerate_dof.py`.

## Configuration

The most important settings live in `Source/config.yaml`:

- `Directory.File_directory`: repository root or any folder containing `Input/` and `Output/`.
- `FEM Parameters.Total Elements`: number of mesh elements.
- `Material Properties`: bulk and shear modulus values used in the forward run.
- `PINN.Type`: switch between forward and inverse usage.
- `Energy.Method`: current energy model (`Neo Hooke` or `Interpolationbasedenergy`).
- `General.Number Load Steps`: number of load increments used in the inverse training loop.

The repository now uses a relative base directory (`..`) so the code is portable across machines.

## Environment setup

This project uses Python with PyTorch, NumPy, and PyYAML.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already have an environment, install the dependencies directly with `pip install -r requirements.txt`.

## How to run

1. Place your mesh and load-step data under `Input/Element512/`.
2. Confirm `Source/config.yaml` points to the correct repository root.
3. Run the forward model:

```bash
python Source/pinn_model.py
```

4. Run the inverse model:

```bash
python Source/pinn_model_inverse.py
```

## Outputs

The training scripts write results to `Output/Element512/`, including:

- epoch-wise loss histories,
- residual histories,
- trained model checkpoints (`.pth`), and
- optional visualization files such as `.vtu`.

## Notes before publishing

- Remove or replace any private experimental data before pushing public changes.
- Check that large binary outputs are ignored or excluded from version control.
- Consider adding your paper title, authors, and a permanent citation if this work is published.
- The repository name currently contains the typo `Paramter`; if you want a cleaner public identity, consider renaming it to `Parameter`.

## Citation

If you use this repository in academic work, you can cite it as software:

```bibtex
@software{rejikumar_2026_pinn_hyperelasticity,
	author  = {Rejikumar, Anandakrishnan},
	title   = {Physics-Informed Neural Networks for Hyperelasticity and Inverse Parameter Identification},
	year    = {2026},
	url     = {https://github.com/anandakrishnanrejikumar/Physics-Informed-Neural-Networks-Meet-Hyperelasticity-and-Inverse-Paramter-Identification}
}
```

## License

See `LICENSE` for the terms of use.
