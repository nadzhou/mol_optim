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
