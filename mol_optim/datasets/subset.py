"""A dataset filtered to one set of elements, written as its own checkpoint.

The action space can only add the elements in `cfg.atom_types`, so a compound carrying a
fluorine the seed does not have is not a hard target — it is an impossible one, and
counting it in a recovery rate makes the rate meaningless. `configs/reachable.toml`
measured that: 380 of seed 0's 565 held-out analogs were unreachable at any depth.

Restricting the *data* instead of widening the action space is the other way to close that
gap, and it is the cleaner experiment. On a C/H/N/O dataset with `atom_types = ["C", "O",
"N"]` no target is blocked by composition, so recovery becomes a measurement of search
alone. It is not free: EGFR loses 70 per cent of its compounds, and the regressor is
refitted on what is left.

This writes a new SDF rather than filtering at load time. One path in the config file is
then the single thing that says which dataset a run used, and no caller can read the
filtered set for one purpose and the full set for another.
"""

from rdkit import Chem

from mol_optim import config
from mol_optim.chem import molio

# Hydrogen is in the element set because it is implicit in an RDKit graph — a molecule of
# carbon and hydrogen alone reports {"C"}. Naming it is what stops "C, H, N, O" from
# looking like it excludes something it does not.
PROPERTIES = ("pic50", "num_measurements", "pic50_spread")


def run(settings: config.Settings) -> None:
    spec = settings.subset
    allowed = set(spec.elements)
    named = molio.read_named(spec.source)
    kept = {
        name: mol
        for name, mol in named.items()
        if {atom.GetSymbol() for atom in mol.GetAtoms()} <= allowed
    }
    print(
        f"{spec.source} -> {spec.path}\n"
        f"{len(kept)} of {len(named)} compounds are {', '.join(sorted(allowed))} only "
        f"({100 * len(kept) / len(named):.1f}%)"
    )
    molecules = tuple(kept.values())
    molio.write(
        spec.path,
        molecules,
        {
            name: [mol.GetProp(name) for mol in molecules] for name in PROPERTIES
        },
    )
    # The names are stereo hashes and everything downstream keys on them, so a molecule
    # that changed name in transit would silently become a different compound.
    back = molio.read_named(spec.path)
    assert set(back) == set(kept), "a compound was renamed on the way to disk"
