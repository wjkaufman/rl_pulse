import qutip as qt
import numpy as np
from scipy.spatial.transform import Rotation


# define system

def get_Hsys(N, cs_strength=1, offset=0, dipolar_strength=1e2,
             rng=None, return_all=False):
    """
    Get system Hamiltonian, defaults to strongly-coupled spin system. Units
    are normalized by the CS standard deviation (line width).
    
    Args:
        cs_strength: Standard deviation of chemical shift strengths.
        dipolar_strength: Standard deviation of dipolar coupling strengths.
    """
    if rng is None:
        rng = np.random.default_rng()
    chemical_shifts = rng.normal(scale=cs_strength, size=(N,))
    offset = rng.normal(scale=offset)
    Hcs = sum(
        [qt.tensor(
            [qt.identity(2)] * i
            + [(offset + chemical_shifts[i]) * qt.sigmaz()]
            + [qt.identity(2)] * (N - i - 1)
        ) for i in range(N)]
    )
    # dipolar interactions
    dipolar_matrix = rng.normal(scale=dipolar_strength, size=(N, N))
    Hdip = sum([
        dipolar_matrix[i, j] * (
            2 * qt.tensor(
                [qt.identity(2)] * i
                + [qt.sigmaz()]
                + [qt.identity(2)] * (j - i - 1)
                + [qt.sigmaz()]
                + [qt.identity(2)] * (N - j - 1)
            )
            - qt.tensor(
                [qt.identity(2)] * i
                + [qt.sigmax()]
                + [qt.identity(2)] * (j - i - 1)
                + [qt.sigmax()]
                + [qt.identity(2)] * (N - j - 1)
            )
            - qt.tensor(
                [qt.identity(2)] * i
                + [qt.sigmay()]
                + [qt.identity(2)] * (j - i - 1)
                + [qt.sigmay()]
                + [qt.identity(2)] * (N - j - 1)
            )
        )
        for i in range(N) for j in range(i + 1, N)
    ])
    if return_all:
        return Hcs + Hdip, (chemical_shifts, offset, dipolar_matrix)
    else:
        return Hcs + Hdip


def get_collective_spin(N):
    X = sum(
        [qt.tensor(
            [qt.identity(2)] * i
            + [qt.spin_Jx(1 / 2)]
            + [qt.identity(2)] * (N - i - 1)
        ) for i in range(N)]
    )
    Y = sum(
        [qt.tensor(
            [qt.identity(2)] * i
            + [qt.spin_Jy(1 / 2)]
            + [qt.identity(2)] * (N - i - 1)
        ) for i in range(N)]
    )
    Z = sum(
        [qt.tensor(
            [qt.identity(2)] * i
            + [qt.spin_Jz(1 / 2)]
            + [qt.identity(2)] * (N - i - 1)
        ) for i in range(N)]
    )
    return (X, Y, Z)

# pulses, pulse names, and corresponding rotations


def get_pulses(Hsys, X, Y, Z, pulse_width=1e-4, delay=1e-3,
               rot_error=0, phase_transient=0, rng=None):
    """
    Args:
        rot_error: Percent error for rotations (consistent errors
            for each pulse).
        phase_transient: Normalized magnitude of phase transient for pulses
            (1 is a full pi/2 pulse). Default is 0.
    """
    pulses = [
        qt.propagator(Hsys, pulse_width),
        qt.propagator(X * (1 + rot_error)
                      + Hsys * pulse_width / (np.pi / 2), np.pi / 2),
        qt.propagator(-X * (1 + rot_error)
                      + Hsys * pulse_width / (np.pi / 2), np.pi / 2),
        qt.propagator(Y * (1 + rot_error)
                      + Hsys * pulse_width / (np.pi / 2), np.pi / 2),
        qt.propagator(-Y * (1 + rot_error)
                      + Hsys * pulse_width / (np.pi / 2), np.pi / 2),
    ]
    if phase_transient > 0:
        # only implementing delta phase transients
        pulses[1] = (qt.propagator(Y, np.pi / 2 * phase_transient)
                     * pulses[1]
                     * qt.propagator(Y, np.pi / 2 * phase_transient))
        pulses[2] = (qt.propagator(-X, np.pi / 2 * phase_transient)
                     * pulses[2]
                     * qt.propagator(-X, np.pi / 2 * phase_transient))
        pulses[3] = (qt.propagator(-Y, np.pi / 2 * phase_transient)
                     * pulses[3]
                     * qt.propagator(-Y, np.pi / 2 * phase_transient))
        pulses[4] = (qt.propagator(X, np.pi / 2 * phase_transient)
                     * pulses[4]
                     * qt.propagator(X, np.pi / 2 * phase_transient))
    delay_propagator = qt.propagator(Hsys, delay)
    pulses = [delay_propagator * i for i in pulses]
    return pulses


pulse_names = [
    'd', 'x', '-x', 'y', '-y',  # 'z', '-z'
]


def pulse_sequence_string(pulse_sequence):
    """Return a string that correspond to pulse sequence
    """
    pulse_list = ','.join([pulse_names[i] for i in pulse_sequence])
    return pulse_list


def get_pulse_sequence(string):
    """Returns a list of integers for the pulse sequence
    """
    chars = string.split(',')
    pulse_sequence = [pulse_names.index(c) for c in chars]
    return pulse_sequence


def get_propagator(pulse_sequence, pulses):
    propagator = qt.identity(pulses[0].dims[0])
    for p in pulse_sequence:
        propagator = pulses[p] * propagator
    return propagator


rotations = [
    np.eye(3),
    np.round(Rotation.from_euler('x', 90, degrees=True).as_matrix()),
    np.round(Rotation.from_euler('x', -90, degrees=True).as_matrix()),
    np.round(Rotation.from_euler('y', 90, degrees=True).as_matrix()),
    np.round(Rotation.from_euler('y', -90, degrees=True).as_matrix()),
    #     np.round(Rotation.from_euler('z', 90, degrees=True).as_matrix()),
    #     np.round(Rotation.from_euler('z', -90, degrees=True).as_matrix()),
]


def get_rotation(pulse_sequence):
    frame = np.eye(3)
    for p in pulse_sequence:
        frame = rotations[p] @ frame
    return frame


def is_cyclic(pulse_sequence):
    frame = get_rotation(pulse_sequence)
    return (frame == np.eye(3)).all()


def count_axes(pulse_sequence):
    """Count time spent on (x, y, z, -x, -y, -z) axes
    """
    axes_counts = [0] * 6
    frame = np.eye(3)
    for p in pulse_sequence:
        frame = rotations[p] @ frame
        axis = np.where(frame[-1, :])[0][0]
        is_negative = np.sum(frame[-1, :]) < 0
        axes_counts[axis + 3 * is_negative] += 1
    return axes_counts


def is_valid_dd(subsequence, sequence_length):
    """Checks if the pulse subsequence allows for dynamical decoupling of
        dipolar interactions (i.e. equal time spent on each axis)
    """
    axes_counts = count_axes(subsequence)
    (x, y, z) = [axes_counts[i] + axes_counts[i + 3] for i in range(3)]
    # time on each axis isn't more than is allowed for dd
    return (np.array([x, y, z]) <= sequence_length / 3).all()


def is_valid_time_suspension(subsequence, sequence_length):
    """Checks if the pulse subsequence allows for dynamical decoupling of
        all interactions (i.e. equal time spent on each ± axis)
    """
    axes_counts = count_axes(subsequence)
    # time on each axis isn't more than is allowed for dd
    return (np.array(axes_counts) <= sequence_length / 6).all()


def get_valid_time_suspension_pulses(subsequence,
                                     num_pulses,
                                     sequence_length):
    valid_pulses = []
    for p in range(num_pulses):
        if is_valid_time_suspension(subsequence + [p], sequence_length):
            valid_pulses.append(p)
    return valid_pulses


# fidelity calculations

def get_fidelity(pulse_sequence, Utarget, pulses):
    Uexp = qt.identity(Utarget.dims[0])
    for p in pulse_sequence:
        Uexp = pulses[p] * Uexp
    return qt.metrics.average_gate_fidelity(Uexp, Utarget)


def get_mean_fidelity(pulse_sequence, Utarget, pulses_ensemble):
    fidelity = 0
    for pulses in pulses_ensemble:
        fidelity += get_fidelity(pulse_sequence, Utarget, pulses)
    return fidelity / len(pulses_ensemble)

# list of existing pulse sequences


whh4 = [0, 1, 4, 0, 3, 2]

ideal6 = [3, 1, 1, 3, 2, 2]
yxx24 = [
    4, 1, 2, 3, 2, 2, 3, 2, 1, 4, 1, 1,
    3, 2, 1, 4, 1, 1, 4, 1, 2, 3, 2, 2
]
yxx48 = [
    3, 2, 2, 3, 2, 2, 4, 1, 1, 3, 2, 2,
    4, 1, 1, 4, 1, 1, 3, 2, 2, 3, 2, 2,
    4, 1, 1, 3, 2, 2, 4, 1, 1, 4, 1, 1,
    3, 2, 2, 4, 1, 1, 3, 2, 2, 4, 1, 1
]

# brute-force search
bf6 = [1, 1, 3, 1, 1, 3]
bf12 = [1, 1, 4, 1, 1, 4,
        2, 2, 4, 2, 2, 4]
bfr12 = [1, 4, 4, 1, 4, 4,
         1, 3, 3, 1, 3, 3]

# vanilla MCTS search
mcts12_1 = [0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1]
mcts12_2 = [4, 0, 4, 1, 4, 4, 3, 4, 2, 2, 2, 0]
mcts12_3 = [4, 4, 4, 1, 0, 3, 3, 0, 1, 3, 3, 4]
mcts12_4 = [3, 0, 3, 1, 3, 0, 3, 2, 1, 1, 1, 2]
mcts24 = [4, 2, 3, 4, 2, 1, 3, 2, 0, 2, 2, 3,
          4, 0, 3, 1, 2, 1, 3, 4, 1, 1, 1, 2]

# AlphaZero MCTS search
# az1 spends equal time on each axis, is cyclic,
# but has low fidelity. Why??
az1 = [
    0, 3, 3, 1, 1, 2, 3, 4, 0, 1, 3, 3,
    1, 0, 2, 2, 2, 0, 3, 3, 4, 1, 3, 3,
    2, 2, 4, 3, 1, 1, 2, 1, 4, 0, 2, 2,
    3, 2, 0, 1, 2, 1, 1, 4, 4, 0, 0, 3
]
# allegedly has reward of 1.26, not terrible
az2 = [
    1, 2, 0, 1, 3, 1, 1, 1, 4, 4, 0, 0,
    1, 4, 1, 0, 1, 4, 1, 0, 2, 1, 3, 0,
    2, 2, 0, 1, 1, 3, 4, 2, 4, 4, 2, 0,
    1, 0, 0, 2, 4, 3, 2, 0, 1, 1, 0, 4
]
# 2020-01-16
# below are 5 of the best-performing time-suspension
# pulse sequence I found from vanilla MCTS (out of 5000 pulse sequences)
# the rewards for the following pulse sequence are
# 2.24, 2.12, 2.07, 2.00, 1.91. Not terrible, but not good compared to
# yxx48 sequence with reward of 3.45.
mcts48_1 = [
    2, 2, 2, 2, 2, 3, 2, 0, 3, 2, 3, 1,
    4, 4, 4, 3, 3, 2, 3, 1, 2, 2, 2, 4,
    3, 2, 1, 4, 4, 4, 1, 3, 0, 3, 1, 1,
    1, 2, 1, 3, 1, 0, 4, 0, 4, 4, 3, 4
]
mcts48_2 = [
    1, 0, 4, 1, 3, 4, 1, 0, 1, 4, 1, 1,
    0, 1, 3, 1, 0, 4, 4, 0, 2, 3, 3, 4,
    3, 1, 3, 1, 2, 0, 3, 1, 3, 0, 3, 3,
    3, 4, 2, 4, 0, 1, 3, 2, 1, 0, 1, 0
]
mcts48_3 = [
    3, 0, 3, 2, 2, 2, 3, 3, 4, 2, 1, 3,
    2, 1, 2, 2, 1, 0, 3, 1, 2, 1, 4, 1,
    4, 1, 1, 3, 1, 1, 4, 4, 4, 4, 1, 0,
    2, 3, 3, 0, 1, 0, 1, 4, 0, 0, 2, 0
]
mcts48_4 = [
    4, 2, 3, 4, 1, 2, 2, 2, 1, 2, 2, 1,
    0, 4, 4, 1, 2, 2, 0, 1, 4, 4, 3, 1,
    0, 1, 0, 3, 0, 1, 1, 2, 4, 4, 3, 4,
    3, 4, 3, 3, 3, 0, 1, 1, 2, 2, 0, 0
]
mcts48_5 = [
    2, 2, 0, 2, 2, 1, 1, 4, 2, 3, 3, 1,
    0, 2, 0, 4, 0, 0, 2, 4, 4, 0, 2, 0,
    2, 1, 2, 0, 4, 3, 0, 4, 0, 0, 4, 2,
    2, 4, 0, 2, 0, 4, 0, 3, 1, 3, 4, 2
]
# from 2021-01-17 run (many more jobs), there
# actually seems to be some promise with this technique!
mcts48_5 = [  # reward is 3.117526668836672
    3, 0, 2, 2, 3, 4, 0, 3, 2, 3, 2, 3,
    3, 2, 2, 3, 2, 3, 3, 1, 0, 4, 0, 1,
    4, 3, 3, 3, 3, 1, 0, 3, 2, 3, 0, 1,
    3, 0, 0, 3, 0, 3, 0, 0, 1, 0, 0, 0
]
mcts48_6 = [  # 3.145176524409188
    3, 2, 3, 0, 4, 4, 4, 2, 3, 3, 2, 3,
    1, 3, 4, 1, 1, 1, 3, 2, 4, 3, 0, 2,
    2, 2, 1, 2, 0, 1, 1, 1, 2, 3, 1, 3,
    1, 1, 2, 1, 2, 4, 2, 2, 4, 0, 0, 0
]
mcts48_7 = [  # 3.4308794670277973
    2, 4, 3, 2, 3, 2, 4, 3, 0, 3, 3, 2,
    3, 1, 3, 4, 1, 4, 1, 1, 0, 3, 3, 3,
    0, 1, 1, 4, 2, 4, 1, 4, 0, 1, 2, 3,
    1, 3, 1, 0, 1, 2, 0, 3, 0, 3, 3, 0
]

# from 2/5 job
az3 = [  # 2.12 reward, vs 2.38 for yxx48
    2, 3, 3, 3, 3, 2, 3, 2, 4, 2, 4, 1,
    2, 3, 3, 3, 0, 3, 0, 2, 3, 2, 2, 2,
    3, 2, 0, 4, 2, 2, 2, 0, 4, 4, 3, 1,
    3, 4, 4, 0, 0, 0, 2, 2, 2, 0, 3, 2
]

# from 2/18 job
az12 = [  # 5.68 reward, vs 5.6 for yxx24 and 5 for yxx48
    1, 4, 2, 2, 3, 2, 2, 3, 1, 1, 4, 1
]

az12_1 = [  # 5.6578 reward
    3, 1, 4, 4, 2, 4, 1, 4, 4, 2, 3, 3
]


class PulseSequenceConfig(object):
    
    def __init__(self,
                 Utarget,
                 N=3,
                 ensemble_size=3,
                 max_sequence_length=48,
                 dipolar_strength=1e2,
                 pulse_width=1e-5,
                 delay=1e-4,
                 rot_error=1e-2,
                 Hsys_ensemble=None,
                 pulses_ensemble=None,
                 sequence=None,
                 rng=None,
                 save_name=None,
                 ):
        """Create a new pulse sequence config object. Basically a collection
        of everything on the physics side of things that is relevant for
        pulse sequences.
        
        Args:
            rot_error: Standard deviation of rotation error to randomly
                sample from.
            save_name (str): Filename to save the ensemble parameters (chemical shift,
                offset, and dipolar matrices). Defaults to None.
        """
        self.N = N
        self.ensemble_size = ensemble_size
        self.max_sequence_length = max_sequence_length
        self.Utarget = Utarget
        self.dipolar_strength = dipolar_strength
        self.pulse_width = pulse_width
        self.delay = delay
        # create a unique rng for multiprocessing purposes
        self.rng = rng if rng is not None else np.random.default_rng()
        if Hsys_ensemble is None:
            self.Hsys_ensemble = []
            if save_name is not None:
                chemical_shifts = []
                offsets = []
                dipolar_matrices = []
            for _ in range(ensemble_size):
                if save_name is not None:
                    H, (cs, offset, dip) = get_Hsys(
                        N=N,
                        dipolar_strength=dipolar_strength,
                        rng=self.rng, return_all=True)
                    chemical_shifts.append(cs)
                    offsets.append(offset)
                    dipolar_matrices.append(dip)
                else:
                    H = get_Hsys(N=N,
                                 dipolar_strength=dipolar_strength,
                                 rng=self.rng)
                self.Hsys_ensemble.append(H)
        else:
            self.Hsys_ensemble = Hsys_ensemble
        if pulses_ensemble is None:
            if save_name is not None:
                rots = []
            X, Y, Z = get_collective_spin(N)
            self.pulses_ensemble = []
            for H in self.Hsys_ensemble:
                rot = self.rng.normal(scale=rot_error)
                if save_name is not None:
                    rots.append(rot)
                self.pulses_ensemble.append(
                    get_pulses(H, X, Y, Z,
                               pulse_width=pulse_width,
                               delay=delay,
                               rot_error=rot)
                )
        else:
            self.pulses_ensemble = pulses_ensemble
        if save_name is not None:
            chemical_shifts = np.stack(chemical_shifts)
            offsets = np.stack(offsets)
            dipolar_matrices = np.stack(dipolar_matrices)
            rots = np.stack(rots)
            np.savez_compressed(save_name, chemical_shifts=chemical_shifts,
                                offsets=offsets, dipolar_matrices=dipolar_matrices,
                                rots=rots)
        self.num_pulses = len(self.pulses_ensemble[0])
        self.pulse_names = pulse_names
        self.sequence = sequence if sequence is not None else []
    
    def reset(self):
        """Reset the pulse sequence config to an empty pulse sequence
        """
        self.sequence = []
    
    def is_done(self):
        """Return whether the pulse sequence is at or beyond its
        maximum sequence length.
        """
        return len(self.sequence) >= self.max_sequence_length
    
    def apply(self, pulse):
        """Apply a pulse to the current pulse sequence.
        """
        self.sequence.append(pulse)
    
    def clone(self):
        """Clone the pulse sequence config. Objects
        that aren't modified are simply returned as-is.
        """
        return PulseSequenceConfig(
            N=self.N,
            ensemble_size=self.ensemble_size,
            max_sequence_length=self.max_sequence_length,
            Utarget=self.Utarget,
            dipolar_strength=self.dipolar_strength,
            pulse_width=self.pulse_width,
            delay=self.delay,
            Hsys_ensemble=self.Hsys_ensemble,
            pulses_ensemble=self.pulses_ensemble,
            sequence=self.sequence.copy(),
            rng=self.rng,
        )
