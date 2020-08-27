"""
Written by Will Kaufman 2020

Inspired by code written by John Vandermause 2016-2017
"""

import numpy as np
import rl_pulse.spin_simulation as ss


def grapeLZAdapt(
        nop=300,
        tau=.1,
        iterations=100,
        delta_guess=np.linspace(10, -10, 300),
        omega_guess=np.ones((300,)),
        initial_state=np.array([[1, 0], [0, 0]], dtype=np.complex64),
        target_state=np.array([[0, 0], [0, 1]], dtype=np.complex64),
        mag_fac=6.6444e2
):
    """Run the GRAPE algorithm to determine control parameters.
    
    https://www.sciencedirect.com/science/article/abs/pii/S1090780704003696

    Arguments:
        nop: No idea. May be number of operations/time steps?
        tau: No idea.
        iterations: Number of iterations to perform gradient ascent.
        delta_guess: Initial guess for delta_omega.
        omega_guess: No idea.
        initial_state: No idea.
        target_state: No idea.
        mag_fac: No idea.

    Returns:
        fidelity: Measure of operator fidelity.
        delta_omega: No idea.
        fidelity_history: Array of operator fidelities for each iteration.
        mag_fac_final: No idea.
    """
    step_size = tau / nop
    # propagate the states
    grad_x = np.zeros((nop,))
    grad_z = np.copy(grad_x)

    fidelity_history = np.zeros((iterations,))
    lambdas = np.zeros((2, 2, nop), dtype=np.complex64)
    rhos = np.copy(lambdas)

    not_working = True  # TODO what is this??
    while not_working:
        print(f'starting while loop (mag_fac: {mag_fac})')
        delta_omega = np.copy(delta_guess)

        not_working = False

        for i in range(iterations):
            rho_update = np.copy(initial_state)
            C_update = np.copy(target_state)

            # propagate the state
            for n in range(nop):
                lambdas[:, :, n] = C_update

                U_j = ss.get_propagator(
                    delta_omega[n] * ss.z + omega_guess[n] * ss.x,
                    step_size
                )

                U_back = ss.get_propagator(
                    delta_omega[nop - n - 1] * ss.z
                    + omega_guess[nop - n - 1] * ss.x,
                    step_size
                )

                rho_update = U_j @ rho_update @ U_j.T.conj()
                C_update = U_back.T.conj() @ C_update @ U_back

                rhos[:, :, n] = rho_update

            # calculate gradients
            for p in range(nop):
                lam_j = lambdas[:, :, nop - p - 1]
                rho_j = rhos[:, :, p]

                int_x = 1j * step_size * (ss.x @ rho_j - rho_j @ ss.x)
                int_z = 1j * step_size * (ss.z * rho_j - rho_j @ ss.z)

                grad_x[p] = -1 * np.real(np.trace(lam_j.T.conj() @ int_x))
                grad_z[p] = -1 * np.real(np.trace(lam_j.T.conj() @ int_z))

            fidelity = ss.fidelity(target_state, rho_update)
            fidelity_history[i] = fidelity

            # print(
            #     'updating delta_omega, grad_z*mag_fac is:\n'
            #     + f'{grad_z*mag_fac}'
            # )
            delta_omega = delta_omega + grad_z * mag_fac

            if i > 1:
                if fidelity_history[i] < fidelity_history[i - 1]:
                    not_working = True
                    mag_fac = mag_fac * 0.9
                    break
    mag_fac_final = mag_fac

    return fidelity, delta_omega, fidelity_history, mag_fac_final


if __name__ == '__main__':
    output = grapeLZAdapt()
    for result in output:
        print(result)
