"""
Dependency policy: every package in requirements.txt has a lower bound (so features we use exist)
AND an upper bound at the next major version (so a surprise major release can't silently break
the pickled model or the dashboard on the next deploy).
"""

import os

import pytest
from packaging.requirements import Requirement

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def requirements():
    with open(os.path.join(ROOT, 'requirements.txt'), encoding='utf-8') as f:
        return [Requirement(line.strip()) for line in f if line.strip() and not line.startswith('#')]


@pytest.mark.parametrize('req', requirements(), ids=lambda r: r.name)
def test_requirement_has_lower_and_upper_bound(req):
    ops = {spec.operator for spec in req.specifier}
    assert '>=' in ops, f"{req.name} needs a minimum version"
    assert '<' in ops, f"{req.name} needs an upper bound (next major version)"


def test_python_version_file_is_pinned():
    path = os.path.join(ROOT, '.python-version')
    version = open(path, encoding='utf-8').read().strip()
    assert version.count('.') == 2, "Render needs a full x.y.z version"
