"""
Written by Will Kaufman 2020

Inspired by code written by John Vandermause 2016-2017
"""

import numpy as np
from math import ceil
from rl_pulse import spin_simulation as ss
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt


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


def get_propagators(
        dim,
        H_controls,
        controls,
        H_system,
        U_target,
        num_steps,
        m,
        step_size):
    """Calculate the forward and backward propagators at each time step.
    
    Arguments:
        TODO fill here
        X:
        P:
    
    Returns: A tuple
        Xs:
        Ps:
    """
    X = np.eye(dim, dtype=np.complex64)
    P = np.copy(U_target)
    Xs = np.zeros((num_steps, dim, dim), dtype=np.complex64)
    Ps = np.copy(Xs)
    
    for n in range(num_steps):
        
        bi = num_steps - n - 1  # back index
        Ps[bi, ...] = P
        
        # forward/backward Hamiltonians at step n
        H_j = np.copy(H_system)
        H_back = np.copy(H_system)
        for k in range(m):
            H_j = (
                H_j
                + controls[k, n] * H_controls[k, ...]
            )
            H_back = (
                H_back
                + controls[k, bi] * H_controls[k, ...]
            )
        U_j = ss.get_propagator(H_j, step_size)
        U_back = ss.get_propagator(-H_back, step_size)
        
        X = U_j @ X
        P = U_back @ P
        
        Xs[n, ...] = X
    
    return Xs, Ps


def get_gradients(
        m,
        num_steps,
        Xs,
        Ps,
        H_controls,
        step_size,
        ):
    """ Calculate gradients for GRAPE algorithm.
    
    Arguments:
        TODO fill here
    
    Returns (array): The gradients array. Shape is m * num_steps.
    
    """
    gradients = np.zeros((m, num_steps))
    
    for n in range(num_steps):
        X_j = Xs[n, ...]
        P_j = Ps[n, ...]
        
        # compute gradients for each control amplitude
        for k in range(m):
            # (see Khaneja, eq. 31)
            gradients[k, n] = -1 * np.real(np.trace(
                P_j.T.conj() @ (
                    1j * step_size
                    * H_controls[k, ...] @ X_j)
            ) * np.trace(
                X_j.T.conj() @ P_j
            ))
    
    return gradients


def grape(
        dim,
        H_controls,
        controls,
        H_system_generator,
        U_target,
        ensemble_size=10,
        T=1e-5,
        control_lims=None,
        iterations=1000,
        epsilon=1e-2,
        printing=False
        ):
    """Run the GRAPE algorithm to implement a CNOT gate.
    
    Arguments:
        dim: Dimension of Hilbert space.
        H_controls: An array of all m control terms in
            the Hamiltonian. Shape is m * dim^2.
        controls: Initial guesses for control amplitudes. Shape is
            m * num_steps.
        H_system_generator: A function that returns a system Hamiltonian
            corresponding to free evolution. Drawn from a distribution of
            possible Hamiltonian values.
        ensemble_size (integer): Number of different system and control
            permutations to evaluate pulse fidelity.
        U_target: Target propagator for the pulse.
        T: Total pulse duration.
        control_lims: Min and max values for each control. Should be an
            m-length array of (min, max) tuples.
        iterations: Number of gradient updates to perform on control
            amplitudes.
        epsilon: Gradient update step size. Either a scalar or a dictionary
            so that epsilon[iteration] is the gradient update step size.
        printing (bool): Whether to print output and display controls while
            running.
    
    Returns:
        tuple: A tuple containing:
            controls: Control amplitudes. Shape is m * num_steps.
            fidelity_history: Fidelities of pulse. Length is iterations.
        
    """
    m = np.size(controls, axis=0)  # number of controls
    num_steps = np.size(controls, axis=1)
    step_size = T / num_steps
    fidelity_history = np.zeros((iterations,))

    if printing:
        time_vals = np.linspace(0, T, num_steps)
        plt.ion()
        axs = []
        for ax in range(m + 1):
            axs.append(plt.subplot(2, ceil((m+1)/2), ax + 1))
    # use Adam optimizer for faster gradient ascent
    beta1 = 1 - 1e-1
    beta2 = 1 - 1e-4
    grad_moment1 = np.zeros_like(controls)
    grad_moment2 = np.zeros_like(controls)
    for j in range(iterations):
        if printing:
            print(f'on iteration {j}')
        
        gradients = np.zeros_like(controls)
        
        for h in range(ensemble_size):
            H_system = H_system_generator()
            control_multiplier = np.random.normal(loc=1, scale=0.005)
            Xs, Ps = get_propagators(
                dim, H_controls, controls * control_multiplier,
                H_system, U_target,
                num_steps, m, step_size)
            
            gradients += get_gradients(
                m, num_steps, Xs, Ps,
                H_controls, step_size)
            
            X = Xs[-1, ...]
            fidelity_history[j] += ss.fidelity(U_target, X)
        
        # rescale gradients and fidelities
        gradients *= 1. / (ensemble_size)
        fidelity_history[j] *= 1. / (ensemble_size)
        
        # TODO continue with Adam optimizer
        grad_moment1 = (beta1 * grad_moment1
                        + (1 - beta1) * gradients)
        grad_moment2 = (beta2 * grad_moment2
                        + (1 - beta2) * gradients**2)
        # then update gradient!
        
        # add gradients to original controls
        if type(epsilon) is dict:
            if j in epsilon.keys():
                epsilon_val = epsilon[j]
        else:
            epsilon_val = epsilon
        alpha = (epsilon_val
                 * np.sqrt(1 - beta2**(j + 1))
                 / (1 - beta1**(j + 1)))
        candidate = (controls + alpha * grad_moment1
                     / (np.sqrt(grad_moment2) + 1e-8))
        if control_lims is not None:
            for k in range(m):
                candidate = np.clip(candidate, control_lims[k][0],
                                    control_lims[k][1])
        controls = candidate
        if printing:
            print(f'fidelity: {fidelity_history[j]}')
            print(f'controls RMS: {np.sqrt(np.mean(controls ** 2))}')

            for ax in axs:
                ax.clear()

            for k in range(m):
                axs[k].plot(time_vals, controls[k, :])
                axs[k].set_title(f'Control {k}')

            axs[m].plot(fidelity_history)
            axs[m].set_title('Fidelity of pulse')

            plt.draw()
            plt.pause(0.01)

    plt.ioff()
    plt.show()
        
    return controls, fidelity_history


def interpolate_controls(controls, num_steps):
    """
    
    """
    f = interp1d(
        np.linspace(0, 1, controls.shape[1]),
        controls,
        kind=3)
    return f(np.linspace(0, 1, num_steps))


def grape_interpolate(
        dim,
        H_controls,
        controls,
        H_system_generator,
        U_target,
        ensemble_size=10,
        T=1e-5,
        control_lims=None,
        iterations=200,
        epsilon=1e-2,
        printing=False,
        step_schedule=None,
        ):
    """Perform the GRAPE algorithm, but increase the time
    resolution of the control amplitudes periodically
    (and maybe also increase the number of spins in the system)
    so that large changes are made quickly and fine-tuning
    is done with greater resolution and more accurate dynamics.
    
    See `grape` documentation for description of arguments.
    
    Arguments:
        iterations: Iterations per num_steps value. The total number
            of iterations is then iterations * len(step_schedule).
        step_schedule: An array of num_steps for control amplitudes.
            Old control amplitude values are interpolated using the
            `interpolate_controls` function. The schedule should
            increase as iterations increase.
    """
    # TODO make epsilon schedule better for this (don't go back to moving
    # all over the place)
    fidelity_history = []
    for num_steps in step_schedule:
        controls = interpolate_controls(controls, num_steps)
        controls, new_fidelity_history = grape(
            dim,
            H_controls,
            controls,
            H_system_generator,
            U_target,
            ensemble_size=ensemble_size,
            T=T,
            control_lims=control_lims,
            iterations=iterations,
            epsilon=epsilon,
            printing=printing
        )
        fidelity_history = np.concatenate(
            (fidelity_history, new_fidelity_history)
        )
    return controls, fidelity_history


if __name__ == '__main__':
    # output = grape_single_spin(
    #     control_z_initial=np.linspace(10, -10, 102),
    #     control_x_initial=np.ones((102,)),
    #     initial_state=np.array([[1, 0], [0, 0]], dtype=np.complex64),
    #     target_state=np.array([[0, 0], [0, 1]], dtype=np.complex64)
    # )
    # for result in output:
    #     print(result)

    # # two spins: CNOT gate
    # dim = 4
    # num_steps = 250
    #
    # H_controls = np.array([
    #     ss.kron(ss.x, np.eye(2)),
    #     ss.kron(ss.z, np.eye(2)),
    #     ss.kron(np.eye(2), ss.x),
    #     ss.kron(np.eye(2), ss.z),
    # ])
    # controls = np.array([
    #     np.linspace(0, 100, num=num_steps),
    #     np.linspace(-100, 100, num=num_steps),
    #     np.linspace(0, 100, num=num_steps),
    #     np.linspace(-100, 100, num=num_steps),
    # ])
    # # controls = np.zeros((4, num_steps))
    # J = 209.17  # J coupling in Hz
    # H_system = 2 * np.pi * J * ss.kron(ss.z, ss.z)
    # # CNOT gate (seems right... TODO check this)
    # U_target = np.array(
    #     [[1, 0, 0, 0],
    #      [0, 1, 0, 0],
    #      [0, 0, 0, 1],
    #      [0, 0, 1, 0]]
    # )
    # # Implement C_21 C_12 (two CNOT gates)
    # # U_target = np.array(
    # #     [[1, 0, 0, 0],
    # #      [0, 0, 1, 0],
    # #      [0, 0, 0, 1],
    # #      [0, 1, 0, 0]])

    # # single spin: x pulse
    # dim = 2
    # num_steps = 100
    #
    # H_controls = np.array([
    #     ss.x,
    #     ss.y
    # ])
    # controls = np.array([
    #     10 * np.ones((num_steps,)),
    #     10 * np.ones((num_steps,)),
    # ])
    # # controls = np.zeros((4, num_steps))
    # J = 209.17  # J coupling in Hz
    # H_system = np.pi * J / 2 * ss.z
    # U_target = ss.get_rotation(ss.x, np.pi/2)
    
    # # 4 spins: x pulse
    # dim = 2**4
    # num_steps = 50
    # step_schedule = [50, 100, 200, 400]
    #
    # (X, Y, Z) = ss.get_total_spin(4, dim)
    #
    # H_controls = np.array([
    #     X,
    #     Y
    # ])
    # controls = 1e3 * np.array([
    #     # np.ones((num_steps,)),
    #     np.sin(np.linspace(0, 3, num_steps)),
    #     # np.zeros((num_steps,))
    #     np.cos(np.linspace(0, 9, num_steps)),
    # ])
    # coupling = 1e3
    # delta = 5e2
    #
    # def H_system_generator():
    #     r1 = np.random.normal(loc=1, scale=0.005)
    #     r2 = np.random.normal(loc=1, scale=0.005)
    #     _, H_system = ss.get_H(4, dim, coupling * r1, delta * r2)
    #     return H_system
    # U_target = ss.get_rotation(X, np.pi/2)
    #
    # # controls, fidelity_history = grape(
    # #     dim=dim,
    # #     H_controls=H_controls,
    # #     controls=controls,
    # #     H_system_generator=H_system_generator,
    # #     U_target=U_target,
    # #     T=1e-3,
    # #     # control_lims=[(-5000, 5000), (0, 5000)],
    # #     iterations=100,
    # #     epsilon={0: 5e2, 25: 1e2, 50: 1e1},
    # #     printing=True,
    # # )
    #
    # controls, fidelity_history = grape_interpolate(
    #     dim=dim,
    #     H_controls=H_controls,
    #     controls=controls,
    #     H_system_generator=H_system_generator,
    #     U_target=U_target,
    #     T=1e-3,
    #     # control_lims=[(-5000, 5000), (0, 5000)],
    #     iterations=20,
    #     epsilon={0: 5e2, 10: 1e2, 15: 1e1},
    #     printing=True,
    #     step_schedule=step_schedule,
    # )
    #
    # print(controls)
    # print(fidelity_history)
    #
    # # np.savez('controls.npz', controls=controls)
    
    print('done!')
