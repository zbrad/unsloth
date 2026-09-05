# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPython 3.14 widened pickle._Pickler._batch_setitems's signature.

3.14 added a required `obj` parameter (so a mid-dict `save` failure can attach
an "item %r" note naming the container), and `dill.Pickler` inherits that
method unchanged from `pickle._Pickler`. `datasets.utils._dill.Pickler` (as of
datasets 4.3.0, the version this project pins) still overrides it with the
pre-3.14 shape `(self, items)`, so `pickle.py`'s own `save_dict` -- calling
`self._batch_setitems(obj.items(), obj)` -- raises

    TypeError: _batch_setitems() takes 2 positional arguments but 3 were given

the instant anything builds a `datasets.Dataset`: every field is hashed
through this Pickler to compute the fingerprint. Every training path here
does that on the very first `Dataset.from_dict` / `.map()` call.

import_fixes.py is loaded by file spec, as test_dataclass_default_backfill.py
does, because `import unsloth.import_fixes` would run unsloth/__init__.py
first, pulling in torch, numpy and unsloth_zoo.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("dill")
pytest.importorskip("datasets")

REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_FIXES = REPO_ROOT / "unsloth" / "import_fixes.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_import_fixes_dill_arity_under_test", IMPORT_FIXES
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_module()
fix_datasets_dill_batch_setitems_arity = _MODULE.fix_datasets_dill_batch_setitems_arity


@pytest.fixture(autouse=True)
def _restore_datasets_pickler():
    """Undo the patch after each test so tests do not leak state into each other."""
    from datasets.utils._dill import Pickler as DatasetsPickler

    original = DatasetsPickler.__dict__.get("_batch_setitems")
    yield
    if original is not None:
        DatasetsPickler._batch_setitems = original


def test_a_dataset_can_be_built_after_the_fix():
    """The actual regression this exists for: fingerprinting must not raise."""
    from datasets import Dataset

    fix_datasets_dill_batch_setitems_arity()
    ds = Dataset.from_dict({"text": ["a", "b", "c"]})
    assert isinstance(ds._fingerprint, str) and ds._fingerprint


def test_the_fix_is_idempotent():
    """A second call must not double-wrap or raise."""
    from datasets.utils._dill import Pickler as DatasetsPickler

    first = fix_datasets_dill_batch_setitems_arity()
    patched = DatasetsPickler.__dict__.get("_batch_setitems")
    second = fix_datasets_dill_batch_setitems_arity()
    assert second is False
    assert DatasetsPickler.__dict__.get("_batch_setitems") is patched
    # On an interpreter that already needs no fix, `first` is False too --
    # either way the method must be safe to call afterwards.
    del first


def test_key_order_insensitivity_is_preserved():
    """The patch must keep datasets' whole reason for overriding this method:
    a dict pickles identically regardless of the order its keys were built in.
    (Not tested via Dataset.from_dict: column order there is schema, not just a
    pickling artifact, so two differently-ordered dicts legitimately fingerprint
    differently -- this needs the dill-level dumps() the fix actually touches.)
    """
    from datasets.utils._dill import dumps

    fix_datasets_dill_batch_setitems_arity()
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert dumps(a) == dumps(b)


def test_a_pre_314_shaped_override_is_left_alone():
    """If a future datasets release already accepts the wider signature, or this
    runs on an interpreter whose dill._batch_setitems needs no `obj`, the probe
    must not touch it."""
    from datasets.utils._dill import Pickler as DatasetsPickler
    import dill

    def _already_fixed(self, items, obj=None):
        return dill.Pickler._batch_setitems(self, items, obj)

    DatasetsPickler._batch_setitems = _already_fixed
    changed = fix_datasets_dill_batch_setitems_arity()
    assert changed is False
    assert DatasetsPickler.__dict__.get("_batch_setitems") is _already_fixed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
