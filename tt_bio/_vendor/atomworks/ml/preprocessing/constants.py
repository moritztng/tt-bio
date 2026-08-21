import re
from enum import Enum

from tt_bio._vendor.atomworks.enums import ChainType

# Cutoff for the number of residues in a peptide
PEPTIDE_MAX_RESIDUES = 20

# Define the "NA" values ("missing" values) that should be treated as NaN (for Pandas)
# NOTE: By default, "NA" is considered as a missing value by Pandas, which is obviously a problem
NA_VALUES = [
    "",
    " ",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
]

PREPROCESSING_SUPPORTED_CHAIN_TYPES = [
    ChainType.NON_POLYMER,
    ChainType.POLYPEPTIDE_L,
    ChainType.POLYPEPTIDE_D,
    ChainType.DNA,
    ChainType.RNA,
    ChainType.BRANCHED,
    ChainType.DNA_RNA_HYBRID,
]
PREPROCESSING_SUPPORTED_CHAIN_TYPES_INTS = [type.value for type in PREPROCESSING_SUPPORTED_CHAIN_TYPES]

TRAINING_SUPPORTED_CHAIN_TYPES = [
    ChainType.NON_POLYMER,
    ChainType.POLYPEPTIDE_L,
    ChainType.DNA,
    ChainType.RNA,
    ChainType.BRANCHED,
]
TRAINING_SUPPORTED_CHAIN_TYPES_INTS = [type.value for type in TRAINING_SUPPORTED_CHAIN_TYPES]

# Regular expression for PDB IDs
PDB_REGEX = re.compile(r"^[0-9A-Za-z]{4}$")


class ClashSeverity(Enum):
    """
    Enum representing the severity of clashes in a PDB file.
    """

    SEVERE = "severe"  # More than 50% of polymers are clashing
    MODERATE = "moderate"  # Any polymers are clashing
    MILD = "mild"  # Any clashes (polymer or non-polymer)
    NO_CLASH = "no-clash"  # No clashes


# Atomic numbers
OXYGEN_ATOMIC_NUMBER = 8
FLUORINE_ATOMIC_NUMBER = 9

# For building the CellList
CELL_SIZE = 4.5

# Entries to exclude from the dataset
ENTRIES_TO_EXCLUDE_FOR_PRE_PROCESSING = [
    # TODO: Deduplicate this list
    "7bho",  # DNA origami
    "7lhd",
    "6vyr",
    "3zif",
    "7nwh",  # Ribosome
    "6nu3",
    "7tbi",  # Nuclear pore complex
    "7lhd",  # Hetero 180-mer
    "4l3b",  # Hetero 180-mer
    "7bsi",  # Hetero 2,820-mer
    "6nhj",  # Hetero 3,300-mer
    "5j7v",  # Homo 8,280-mer
    "6w19",  # Hetero 3,060-mer
    "6cgr",  # Hetero 2,760-mer
    "6b43",  # Hetero 306-mer
    "6cgr",  # Hetero 2,760-mer
    "7bw6",
    "6lgl",
    "7as5",
    "7fj1",
    "7r5k",
    "3k1q",
    "6ncl",
    "5zap",
    "6b43",
    "1m4x",
    "6q1f",
    "7qiz",
    "4f5x",
    "7tbk",
    "5jus",
    "1uf2",
    "6ftg",
    "7tbk",
    "7v4t",
    "7r5j",
    "3jbp",
    "6ylh",
    "7btb",
    "6zvk",
    "4u3n",
    "6woo",
    "6ftj",
    "4u3n",
    "3jan",
    "4u3n",
    "5tbw",
    "6tb3",
    "6zvk",
    "5dat",
    "3jan",
    "4v5x",
]
