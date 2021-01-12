import qutip as qt
import numpy as np
from scipy.spatial.transform import Rotation


# define system

def get_Hsys(N, dipolar_strength=1e-2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    chemical_shifts = 2 * np.pi * rng.normal(scale=1, size=(N,))
    Hcs = sum(
        [qt.tensor(
            [qt.identity(2)] * i
            + [chemical_shifts[i] * qt.sigmaz()]
            + [qt.identity(2)] * (N - i - 1)
        ) for i in range(N)]
    )
    # dipolar interactions
    dipolar_matrix = 2 * np.pi * \
        rng.normal(scale=dipolar_strength, size=(N, N))
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


def get_pulses(Hsys, X, Y, Z, pulse_width, delay, rot_error=0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    rot = rng.normal(scale=rot_error)
    pulses = [
        qt.propagator(Hsys, pulse_width),
        qt.propagator(X * (np.pi / 2) * (1 + rot)
                      / pulse_width + Hsys, pulse_width),
        qt.propagator(-X * (np.pi / 2) * (1 + rot)
                      / pulse_width + Hsys, pulse_width),
        qt.propagator(Y * (np.pi / 2) * (1 + rot)
                      / pulse_width + Hsys, pulse_width),
        qt.propagator(-Y * (np.pi / 2) * (1 + rot)
                      / pulse_width + Hsys, pulse_width),
        # qt.propagator(Z * (np.pi/2) * (1 + rot)
        #               / pulse_width + Hsys, pulse_width),
        # qt.propagator(-Z * (np.pi/2) * (1 + rot)
        #               / pulse_width + Hsys, pulse_width),
    ]
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


# define a class

class PulseSequenceConfig(object):
    
    def __init__(self,
                 N,
                 ensemble_size,
                 max_sequence_length,
                 Utarget,
                 dipolar_strength=1e-2,
                 pulse_width=1e-3,
                 delay=1e-2,
                 Hsys_ensemble=None,
                 pulses_ensemble=None,
                 sequence=None,
                 propagators=None,
                 frame=None,
                 axis_counts=None,
                 ):
        """Create a new pulse sequence config object. Basically a collection
        of everything on the physics side of things that is relevant for
        pulse sequences.
        
        Args:
            propagators: A list of Qobj that represents the propagators for
                the ensemble for the given pulse sequence.
            frame: A 3x3 matrix representing the collective rotation from the
                pulse sequence.
        """
        self.N = N
        self.ensemble_size = ensemble_size
        self.max_sequence_length = max_sequence_length
        self.Utarget = Utarget
        self.dipolar_strength = dipolar_strength
        self.pulse_width = pulse_width
        self.delay = delay
        # create a unique rng for multiprocessing purposes
        self.rng = np.random.default_rng()
        if Hsys_ensemble is None:
            self.Hsys_ensemble = [
                get_Hsys(N, dipolar_strength=dipolar_strength, rng=self.rng)
                for _ in range(ensemble_size)
            ]
        else:
            self.Hsys_ensemble = Hsys_ensemble
        if pulses_ensemble is None:
            X, Y, Z = get_collective_spin(N)
            self.pulses_ensemble = [
                get_pulses(H, X, Y, Z, pulse_width, delay,
                           rot_error=0.01, rng=self.rng)
                for H in self.Hsys_ensemble
            ]
        else:
            self.pulses_ensemble = pulses_ensemble
        self.num_pulses = len(self.pulses_ensemble[0])
        self.pulse_names = pulse_names
        if sequence is None:
            self.sequence = []
        else:
            self.sequence = sequence
        if propagators is None:
            self.propagators = [qt.identity(Utarget.dims[0])] * ensemble_size
            # what sequence length the propagators correspond to
            self.propagator_seq_length = 0
        else:
            self.propagators = propagators
            self.propagator_seq_length = len(sequence)
        if frame is None:
            self.frame = np.eye(3)
        else:
            self.frame = frame
        if axis_counts is None:
            self.axis_counts = np.zeros((6,))
        else:
            self.axis_counts = axis_counts
    
    def reset(self):
        """Reset the pulse sequence config to an empty pulse sequence
        """
        self.sequence = []
        self.propagators = ([qt.identity(self.Utarget.dims[0])]
                            * self.ensemble_size)
        self.propagator_seq_length = 0
        self.frame = np.eye(3)
        self.axis_counts = np.zeros((6,))
    
    def get_valid_time_suspension_pulses(self):
        valid_pulses = []
        for p in range(len(self.pulses_ensemble[0])):
            new_frame = rotations[p] @ self.frame
            axis = np.where(new_frame[-1, :])[0][0]
            is_negative = np.sum(new_frame[-1, :]) < 0
            new_counts = self.axis_counts.copy()
            new_counts[axis + 3 * is_negative] += 1
            if (new_counts <= self.max_sequence_length / 6).all():
                valid_pulses.append(p)
        return valid_pulses
    
    def get_valid_pulses(self):
        """Return all valid pulses that can be used for time suspension
        """
        return self.get_valid_time_suspension_pulses()
    
    def is_done(self):
        """Return whether the pulse sequence is at or beyond its
        maximum sequence length.
        """
        return len(self.sequence) >= self.max_sequence_length
    
    def apply(self, pulse, update_propagators=False):
        """Apply a pulse to the current pulse sequence.
        """
        self.sequence.append(pulse)
        if update_propagators:
            self.update_propagators()
        self.update_frame(pulse)
    
    def update_propagators(self):
        """Returns the propagator corresponding to the pulse sequence.
        Uses self.propagators to store previous computation and (hopefully)
        speed things up significantly.
        """
        for p in range(self.propagator_seq_length, len(self.sequence)):
            for s in range(self.ensemble_size):
                self.propagators[s] = (
                    self.pulses_ensemble[s][self.sequence[p]]
                    * self.propagators[s])
        self.propagator_seq_length = len(self.sequence)
        return self.propagators
    
    def update_frame(self, pulse):
        """Update frame and axis counts
        """
        self.frame = rotations[pulse] @ self.frame
        axis = np.where(self.frame[-1, :])[0][0]
        is_negative = np.sum(self.frame[-1, :]) < 0
        self.axis_counts[axis + 3 * is_negative] += 1
    
    def get_mean_fidelity(self):
        self.update_propagators()
        fidelity = 0
        for s in range(self.ensemble_size):
            fidelity += np.clip(
                qt.metrics.average_gate_fidelity(
                    self.propagators[s],
                    self.Utarget),
                0, 1
            )
        return fidelity / self.ensemble_size
    
    def value(self):
        """Return the value (or 'reward') of the pulse sequence,
        defined as minus log infidelity.
        """
        fidelity = self.get_mean_fidelity()
        return -1.0 * np.log10(1 - fidelity + 1e-200)
    
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
            propagators=self.propagators,
            frame=self.frame.copy(),
            axis_counts=self.axis_counts.copy(),
        )
