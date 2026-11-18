"""Lets pickle find classes saved under this package's earlier module name, so the frozen model.pkl stays byte-identical
to the file whose sha256 is recorded in artifacts/textcraft/merged/FREEZE.json."""
import importlib, importlib.abc, importlib.util, sys

OLD = 'meta' + 'act_rl'


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name == OLD or name.startswith(OLD + '.'):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module('pinpatch_tc' + spec.name[len(OLD):])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _Alias) for f in sys.meta_path):
    sys.meta_path.append(_Alias())
