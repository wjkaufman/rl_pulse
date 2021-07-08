# RL-Pulse

What's the best way to implement high-fidelity unitary operators on a quantum system? Given a target effective Hamiltonian, what are good ways to achieve the target Hamiltonian (taking into account interactions between terms of the Hamiltonian, experimental constraints, and errors)? How do you even know if the operation is high-fidelity? These are some of the questions we're trying to answer.

This repository contains Python code to simulate unitary dynamics for a spin system in a magnetic field (see `spin_simulation`), as well as a mess of code to apply reinforcement learning algorithms to the Hamiltonian engineering problem.

## Getting pulse sequences from data

Running `run_alpha_zero` saves a job directory with tensorboard data and network parameters saved periodically during training. To see the tensorboard data, run `tensorboard --logdir .` in the directory containing the job files. Also, candidate pulse sequences are written to standard output, which is saved in files that are read later.

My ad-hoc process for pulling pulse sequences uses `Get-Candidate-Sequences` (in `rl_pulse/eval/`). It has more detail there.

## Documentation

Using Sphinx, `sphinx-apidocs`, and `autodoc` to create API docs from docstrings. I need to make sure I've fully documented the code though... And find a place to put the documentation once it's built.

To generate `rst` files with `sphinx-apidocs`, run `sphinx-apidoc -o docs/ .` from the root directory.

## Citation

If you use this code or any ideas presented in the repository, please cite it as

```
@misc{RLPulse,
  title = {{RL-Pulse}: A library for Hamiltonian Engineering using Reinforcement Learning},
  author = "{Will Kaufman and Chandrasekhar Ramanathan}",
  howpublished = {\url{https://github.com/wjkaufman/rl_pulse}},
  url = "https://github.com/wjkaufman/rl_pulse",
  year = 2020,
  note = "[Online; accessed 01-May-2020]"
}
```
