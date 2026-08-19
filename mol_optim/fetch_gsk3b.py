"""Turns TDC's published GSK3B oracle into the arrays oracle_gsk3b.py reads. Run once:

    python -m mol_optim.fetch_gsk3b

The published file is a 28 MB pickle of a scikit-learn RandomForestClassifier. It is
read here exactly once, and what this writes is a 2 MB npz of plain integer and float
arrays — no scikit-learn at runtime, no version of it that the file has to match, and
nothing to unpickle again.

Nothing under sklearn is imported or run. The unpickler below maps every sklearn class
to an inert stub that just collects the state dict it is handed, so what comes back is
numpy arrays and stub objects holding them. That is also what makes unpickling a file
off the internet safe here: a pickle can only construct what find_class hands back.
"""

import hashlib
import pickle
import urllib.request

import numpy as np

from mol_optim import oracle_gsk3b

# tdc.metadata.oracle2id["gsk3b_current"] — the file tdc.Oracle("GSK3B") downloads and
# loads on any scikit-learn newer than 0.24, which today means all of them.
PICKLE_URL = "https://dataverse.harvard.edu/api/access/datafile/6413412"
PICKLE_SHA256 = "d3a20701b80e5179c88c3ad4dc3483dd7ab35c50dc055c6773a7f5b63e89b6d5"
PICKLE_PATH = oracle_gsk3b.MODEL_PATH.with_name("gsk3b_current.pkl")


class Stub:
    """Stands in for an sklearn object: takes any constructor, keeps whatever state."""

    def __init__(self, *args):
        self.constructor_args = args


class StubUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("sklearn"):
            return type(name, (Stub,), {})
        return super().find_class(module, name)


if __name__ == "__main__":
    if not PICKLE_PATH.exists():
        PICKLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {PICKLE_URL}")
        urllib.request.urlretrieve(PICKLE_URL, PICKLE_PATH)
    digest = hashlib.sha256(PICKLE_PATH.read_bytes()).hexdigest()
    if digest != PICKLE_SHA256:
        raise ValueError(
            f"{PICKLE_PATH} hashes to {digest}, not the pinned {PICKLE_SHA256}. "
            "The upstream file changed; re-check it against TDC before trusting it."
        )

    with open(PICKLE_PATH, "rb") as pickle_file:
        forest = StubUnpickler(pickle_file).load()
    if list(forest.classes_) != [0.0, 1.0]:
        raise ValueError(f"expected classes [0, 1], got {forest.classes_}")

    # One node table for all 100 trees. Child indices are rewritten to point into the
    # concatenation, so a walk never needs to know which tree it is in.
    bit, left, right, probability, roots = [], [], [], [], []
    offset = 0
    for tree in (estimator.tree_ for estimator in forest.estimators_):
        nodes = tree.nodes
        is_leaf = nodes["left_child"] == -1  # sklearn's TREE_LEAF
        if not np.all(nodes["threshold"][~is_leaf] == 0.5):
            raise ValueError(
                "a split is not a bit test: the features this forest was fitted on "
                "were not 0/1 fingerprint bits, so oracle_gsk3b.score cannot walk it"
            )

        # A leaf points at itself and tests bit 0, so a walk that arrives sits still.
        self_index = np.arange(tree.node_count) + offset
        bit.append(np.where(is_leaf, 0, nodes["feature"]).astype(np.int16))
        left.append(
            np.where(is_leaf, self_index, nodes["left_child"] + offset).astype(np.int32)
        )
        right.append(
            np.where(is_leaf, self_index, nodes["right_child"] + offset).astype(np.int32)
        )
        # Class counts at the leaf, as a probability — what predict_proba averages.
        counts = tree.values[:, 0, :]  # [node_count, 2]
        probability.append((counts[:, 1] / counts.sum(axis=1)).astype(np.float32))
        roots.append(offset)
        offset += tree.node_count

    depth = max(tree.max_depth for tree in (e.tree_ for e in forest.estimators_))
    oracle_gsk3b.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        oracle_gsk3b.MODEL_PATH,
        bit=np.concatenate(bit),
        left=np.concatenate(left),
        right=np.concatenate(right),
        probability=np.concatenate(probability),
        roots=np.array(roots, dtype=np.int32),
        depth=depth,
    )
    written = oracle_gsk3b.MODEL_PATH
    print(
        f"wrote {written} — {len(roots)} trees, {offset} nodes, depth {depth}, "
        f"{written.stat().st_size / 1e6:.1f} MB"
    )
