#!/usr/bin/env python

import math

# Some constants for use in the Tm calculation function

# Constants for SantaLucia melting temperature.
ENTHALPY_VALUES = {
    'AA': -7.9,
    'AC': -8.4,
    'AG': -7.8,
    'AT': -7.2,
    'CA': -8.5,
    'CC': -8,
    'CG': -10.6,
    'CT': -7.8,
    'GA': -8.2,
    'GC': -9.8,
    'GG': -8,
    'GT': -8.4,
    'TA': -7.2,
    'TC': -8.2,
    'TG': -8.5,
    'TT': -7.9,
    'A': 2.3,
    'C': 0.1,
    'G': 0.1,
    'T': 2.3,
}

ENTROPY_VALUES = {
    'AA': -22.2,
    'AC': -22.4,
    'AG': -21,
    'AT': -20.4,
    'CA': -22.7,
    'CC': -19.9,
    'CG': -27.2,
    'CT': -21,
    'GA': -22.2,
    'GC': -24.4,
    'GG': -19.9,
    'GT': -22.4,
    'TA': -21.3,
    'TC': -22.2,
    'TG': -22.7,
    'TT': -22.2,
    'A': 4.1,
    'C': -2.8,
    'G': -2.8,
    'T': 4.1,
}

COMPLEMENT_NUCS = {
    'A': 'T',
    'T': 'A',
    'C': 'G',
    'G': 'C'
}

SL_SYMMETRY_CORRECTION = 1.38
SL_SALT_CORRECTION = 0.368
MOLAR_GAS_CONSTANT = 1.98722
ABS_ZERO_CELSIUS = -273.15

def calc_tm_sl(seq, dna_conc, salt_conc):
    """
    Calculates the melting temperature using the AmpliExpress
    implementation of the SantaLucia algorithm.

    :param seq: the sequence for which to measure the melting
        temperature
    :param dna_conc: the molar concentration of the DNA
    :param salt_conc: the molar concentration of salt (sodium, Na)
    :returns: the melting temperature of the ``seq``

    """
    seq = seq.upper()

    # Set initiation entropy
    enthalpy = ENTHALPY_VALUES[seq[0]]
    entropy = ENTROPY_VALUES[seq[0]]

    # Add up the entropy for the remainder of the sequence, including
    # the final nucleotide.
    for i in range(len(seq)):
        subseq = seq[i:i+2]
        enthalpy += ENTHALPY_VALUES[subseq]
        entropy += ENTROPY_VALUES[subseq]

    #print enthalpy, entropy
    # Symmetry correction
    entropy -= SL_SYMMETRY_CORRECTION
    # Salt correction of entropy
    entropy += SL_SALT_CORRECTION * len(seq) * math.log(salt_conc)

    #print entropy
    #print len(seq)
    #print math.log(salt_conc)
    #print dna_conc
    #print math.log(dna_conc)
    # Compute Tm based on entropy and enthalpy
    melting_temp = (
        (enthalpy * 1000 /
         (entropy + MOLAR_GAS_CONSTANT * math.log(dna_conc))) +
        ABS_ZERO_CELSIUS
    )

    # Compute Gibs free energy
    gibs = (enthalpy * 1000 - (melting_temp - ABS_ZERO_CELSIUS) * entropy) / 1000

    return {"Tm": melting_temp, "Gibs": gibs} 



if __name__ == "__main__":
    seq = 'TTTGGAGCCTGGATGGGAAGCAGTGTGCAC'
    dna_conc = 0.00000025
    salt_conc = 0.05
    Tm = calc_tm_sl(seq, dna_conc, salt_conc)
    print (seq, Tm)
