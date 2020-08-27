"""
Written by Will Kaufman 2020

Inspired by code written by John Vandermause 2016-2017
"""

import numpy as np
from rl_pulse import spin_simulation as ss


def grape_single_spin(
        control_z_initial,
        control_x_initial,
        initial_state,
        target_state,
        num_steps=102,
        tau=1e-1,
        iterations=100,
        epsilon=1e4
):
    """Run the GRAPE algorithm to optimize control amplitudes.
    
    https://www.sciencedirect.com/science/article/abs/pii/S1090780704003696

    Arguments:
        num_steps: Number of steps that the control amplitudes are defined for.
        tau: Total time for the GRAPE pulse.
        iterations: Number of iterations to perform gradient ascent.
        control_z_initial: Initial control amplitudes for B_z.
        control_x_initial: Initial control amplitudes for B_x.
        initial_state: Initial density operator for the system.
        target_state: Target density operator for the system.
        epsilon: Step size for gradient update.

    Returns:
        fidelity: Measure of operator fidelity.
        control_z: No idea.
        fidelity_history: Array of operator fidelities for each iteration.
        epsilon_final: No idea.
    """
    time_step = tau / num_steps
    # propagate the states
    grad_x = np.zeros((num_steps,))
    grad_z = np.copy(grad_x)

    fidelity_history = np.zeros((iterations,))
    lambdas = np.zeros((2, 2, num_steps), dtype=np.complex64)
    rhos = np.copy(lambdas)

    not_working = True  # TODO what is this??
    while not_working:
        print(f'starting while loop (epsilon: {epsilon})')
        control_z = np.copy(control_z_initial)
        control_x = np.copy(control_x_initial)

        not_working = False

        for i in range(iterations):
            rho_update = np.copy(initial_state)
            C_update = np.copy(target_state)

            # propagate the state
            for n in range(num_steps):
                lambdas[:, :, n] = C_update

                U_j = ss.get_propagator(
                    control_z[n] * ss.z + control_x[n] * ss.x,
                    time_step
                )

                U_back = ss.get_propagator(
                    control_z[num_steps - n - 1] * ss.z
                    + control_x[num_steps - n - 1] * ss.x,
                    time_step
                )

                rho_update = U_j @ rho_update @ U_j.T.conj()
                C_update = U_back.T.conj() @ C_update @ U_back

                rhos[:, :, n] = rho_update

            # calculate gradients
            for p in range(num_steps):
                lambda_j = lambdas[:, :, num_steps - p - 1]
                rho_j = rhos[:, :, p]

                # define commutators [H_k, rho_j] (Khaneja 2005, eq. 12)
                int_x = 1j * time_step * (ss.x @ rho_j - rho_j @ ss.x)
                int_z = 1j * time_step * (ss.z * rho_j - rho_j @ ss.z)

                grad_x[p] = -1 * np.real(np.trace(lambda_j.T.conj() @ int_x))
                grad_z[p] = -1 * np.real(np.trace(lambda_j.T.conj() @ int_z))

            fidelity = ss.fidelity(target_state, rho_update)
            fidelity_history[i] = fidelity

            control_z = control_z + grad_z * epsilon
            control_x = control_x + grad_x * epsilon

            if i > 1:
                if fidelity_history[i] < fidelity_history[i - 1]:
                    not_working = True
                    epsilon = epsilon * 0.9
                    break
    epsilon_final = epsilon

    return fidelity, control_z, control_x, fidelity_history, epsilon_final


if __name__ == '__main__':
    output = grape_single_spin(
        control_z_initial=np.linspace(10, -10, 102),
        control_x_initial=np.ones((102,)),
        initial_state=np.array([[1, 0], [0, 0]], dtype=np.complex64),
        target_state=np.array([[0, 0], [0, 1]], dtype=np.complex64)
    )
    for result in output:
        print(result)
