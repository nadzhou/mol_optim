"""Reading the ZINC file. The download is pinned by hash; this is the parse below it.

Written against temporary files rather than the real 12 MB one, so it runs on a checkout
that has not downloaded anything. `test_pretrain.py` covers the real file.
"""

from pathlib import Path

import pytest

from mol_optim.datasets import zinc


def tab(tmp_path: Path, body: str, header: str = "smiles") -> Path:
    path = tmp_path / "zinc.tab"
    path.write_text(f"{header}\n{body}")
    return path


def test_a_file_with_other_columns_is_refused(tmp_path):
    # The pinned file has one column. A different shape means a different file, and
    # column 0 of it is not necessarily a SMILES.
    path = tab(tmp_path, "C\n", header="smiles\tzinc_id")
    with pytest.raises(ValueError, match="single 'smiles' column"):
        zinc.molecules(path)


def test_an_unreadable_record_stops_the_load(tmp_path):
    # Skipping it silently would change which molecules the encoder was pretrained on
    # without changing the pinned hash's story about it.
    path = tab(tmp_path, "CCO\nnot-a-molecule\n")
    with pytest.raises(ValueError, match="could not read"):
        zinc.molecules(path)