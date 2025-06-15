# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
from typing import Tuple, Union, Dict, Any

from spack.version import GitVersion, StandardVersion


VersionType = Union[GitVersion, StandardVersion]


def concretization_version_order(
    version_info: Tuple[VersionType, Dict[str, Any]]
) -> Tuple[bool, bool, bool, bool, VersionType]:
    """Version order key for concretization, where preferred > not preferred,
    not deprecated > deprecated, finite > any infinite component; only if all are
    the same, do we use default version ordering."""
    version: VersionType
    info: Dict[str, Any]
    version, info = version_info
    is_preferred: bool = info.get("preferred", False)
    is_not_deprecated: bool = not info.get("deprecated", False)
    is_not_develop: bool = not version.isdevelop()
    is_not_prerelease: bool = not version.is_prerelease()
    return (
        is_preferred,
        is_not_deprecated,
        is_not_develop,
        is_not_prerelease,
        version,
    )
