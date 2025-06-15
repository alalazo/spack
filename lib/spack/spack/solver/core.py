# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Low-level wrappers around clingo API."""
import importlib
import pathlib
from types import ModuleType
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union, Dict

from llnl.util import lang


def _ast_getter(*names: str) -> Callable[[Any], Any]:
    """Helper to retrieve AST attributes from different versions of the clingo API"""

    def getter(node: Any) -> Any:
        for name in names:
            result: Optional[Any] = getattr(node, name, None)
            if result:
                return result
        raise KeyError(f"node has no such keys: {names}")

    return getter


ast_type: Callable[[Any], Any] = _ast_getter("ast_type", "type")
ast_sym: Callable[[Any], Any] = _ast_getter("symbol", "term")


class AspObject:
    """Object representing a piece of ASP code."""
    pass


def _id(thing: Any) -> Union[str, int, "AspObject"]:
    """Quote string if needed for it to be a valid identifier."""
    if isinstance(thing, bool):
        return f'"{thing}"'
    elif isinstance(thing, (AspObject, int)):
        return thing
    else:
        str_thing: str
        if isinstance(thing, str):
            # escape characters that cannot be in clingo strings
            str_thing = thing.replace("\\", r"\\")
            str_thing = str_thing.replace("\n", r"\n")
            str_thing = str_thing.replace('"', r"\"")
        else:
            str_thing = str(thing)
        return f'"{str_thing}"'


class AspVar(AspObject):
    """Represents a variable in an ASP rule, allows for conditionally generating
    rules"""

    def __init__(self, name: str) -> None:
        self.name: str = name

    def __str__(self) -> str:
        return str(self.name)


@lang.key_ordering
class AspFunction(AspObject):
    """A term in the ASP logic program"""

    __slots__ = ("name", "args")

    name: str
    args: Tuple[Any, ...]

    def __init__(self, name: str, args: Optional[Tuple[Any, ...]] = None) -> None:
        self.name = name
        self.args = () if args is None else tuple(args)

    def _cmp_key(self) -> Tuple[str, Tuple[Any, ...]]:
        return self.name, self.args

    def __call__(self, *args: Any) -> "AspFunction":
        """Return a new instance of this function with added arguments.

        Note that calls are additive, so you can do things like::

            >>> attr = AspFunction("attr")
            attr()

            >>> attr("version")
            attr("version")

            >>> attr("version")("foo")
            attr("version", "foo")

            >>> v = AspFunction("attr", "version")
            attr("version")

            >>> v("foo", "bar")
            attr("version", "foo", "bar")

        """
        return AspFunction(self.name, self.args + args)

    def __str__(self) -> str:
        args_str: str = f"({','.join(str(_id(arg)) for arg in self.args)})"
        return f"{self.name}{args_str}"

    def __repr__(self) -> str:
        return str(self)


class _AspFunctionBuilder:
    def __getattr__(self, name: str) -> AspFunction:
        return AspFunction(name)


#: Global AspFunction builder
fn: _AspFunctionBuilder = _AspFunctionBuilder()

_CLINGO_MODULE: Optional[ModuleType] = None


def clingo() -> ModuleType:
    """Lazy imports the Python module for clingo, and returns it."""
    global _CLINGO_MODULE
    if _CLINGO_MODULE is not None:
        return _CLINGO_MODULE

    clingo_mod_local: Optional[ModuleType] = None
    try:
        clingo_mod_local = importlib.import_module("clingo")
        # Make sure we didn't import an empty module
        _ensure_clingo_or_raise(clingo_mod_local)
    except ImportError:
        pass  # Handled below

    if clingo_mod_local is not None:
        _CLINGO_MODULE = _set_clingo_module_cache(clingo_mod_local)
        return _CLINGO_MODULE

    clingo_mod_local = _bootstrap_clingo()
    _CLINGO_MODULE = _set_clingo_module_cache(clingo_mod_local)
    return _CLINGO_MODULE


def _set_clingo_module_cache(clingo_mod: ModuleType) -> ModuleType:
    """Sets the global cache to the lazy imported clingo module"""
    # global _CLINGO_MODULE # Removed as it's already handled in clingo()
    importlib.import_module("clingo.ast")
    # _CLINGO_MODULE = clingo_mod # This line is redundant, _CLINGO_MODULE is set in clingo()
    return clingo_mod


def _ensure_clingo_or_raise(clingo_mod: Optional[ModuleType]) -> None:
    """Ensures the clingo module can access expected attributes, otherwise raises an error."""
    # These are imports that may be problematic at top level (circular imports). They are used
    # only to provide exhaustive details when erroring due to a broken clingo module.
    import spack.config
    import spack.paths as sp_paths # Renamed to avoid conflict
    import spack.util.path as sup

    if clingo_mod is None:
        raise RuntimeError("Clingo module not found. Please ensure it is installed or bootstrapped.")

    try:
        clingo_mod.Symbol # type: ignore[attr-defined]
    except AttributeError:
        assert clingo_mod.__file__ is not None, "clingo installation is incomplete or invalid"
        # Reaching this point indicates a broken clingo installation
        # If Spack derived clingo, suggest user re-run bootstrap
        # if non-spack, suggest user investigate installation
        # assume Spack is not responsible for broken clingo
        msg: str = (
            f"Clingo installation at {clingo_mod.__file__} is incomplete or invalid."
            "Please repair installation or re-install. "
            "Alternatively, consider installing clingo via Spack."
        )
        # check whether Spack is responsible
        if (
            pathlib.Path(
                sup.canonicalize_path(
                    spack.config.CONFIG.get("bootstrap:root", sp_paths.default_user_bootstrap_path)
                )
            )
            in pathlib.Path(clingo_mod.__file__).parents
        ):
            # Spack is responsible for the broken clingo
            msg = (
                "Spack bootstrapped copy of Clingo is broken, "
                "please re-run the bootstrapping process via command `spack bootstrap now`."
                " If this issue persists, please file a bug at: github.com/spack/spack"
            )
        raise RuntimeError(
            "Clingo installation may be broken or incomplete, "
            "please verify clingo has been installed correctly"
            "\n\nClingo does not provide symbol clingo.Symbol. " # Added space
            f"{msg}"
        )


def clingo_cffi() -> bool:
    """Returns True if clingo uses the CFFI interface"""
    return hasattr(clingo().Symbol, "_rep") # type: ignore[attr-defined]


def _bootstrap_clingo() -> ModuleType:
    """Bootstraps the clingo module and returns it"""
    import spack.bootstrap

    with spack.bootstrap.ensure_bootstrap_configuration():
        spack.bootstrap.ensure_clingo_importable_or_raise()
        clingo_mod_boot: ModuleType = importlib.import_module("clingo")

    return clingo_mod_boot


def parse_files(*args: Any, **kwargs: Any) -> Any:
    """Wrapper around clingo parse_files, that dispatches the function according
    to clingo API version.
    """
    clingo_api: ModuleType = clingo()
    try:
        return importlib.import_module("clingo.ast").parse_files(*args, **kwargs)
    except (ImportError, AttributeError):
        return clingo_api.parse_files(*args, **kwargs) # type: ignore[attr-defined]


def parse_term(*args: Any, **kwargs: Any) -> Any:
    """Wrapper around clingo parse_term, that dispatches the function according
    to clingo API version.
    """
    clingo_api: ModuleType = clingo()
    try:
        return importlib.import_module("clingo.symbol").parse_term(*args, **kwargs)
    except (ImportError, AttributeError):
        return clingo_api.parse_term(*args, **kwargs) # type: ignore[attr-defined]


class NodeArgument(NamedTuple):
    """Represents a node in the DAG"""

    id: str
    pkg: str


class NodeFlag(NamedTuple):
    flag_type: str
    flag: str
    flag_group: str
    source: str


def intermediate_repr(sym: Any) -> Union[str, "NodeArgument", "NodeFlag", Tuple[Any, ...]]:
    """Returns an intermediate representation of clingo models for Spack's spec builder.

    Currently, transforms symbols from clingo models either to strings or to NodeArgument objects.

    Returns:
        This will turn a ``clingo.Symbol`` into a string or NodeArgument, or a sequence of
        ``clingo.Symbol`` objects into a tuple of those objects.
    """
    # TODO: simplify this when we no longer have to support older clingo versions.
    if isinstance(sym, (list, tuple)):
        return tuple(intermediate_repr(a) for a in sym) # type: ignore[no-any-return]

    try:
        # It's assumed sym has 'name' and 'arguments' if it's a clingo Function symbol
        if sym.name == "node":
            return NodeArgument(
                id=str(intermediate_repr(sym.arguments[0])), # Ensure id is str
                pkg=str(intermediate_repr(sym.arguments[1]))  # Ensure pkg is str
            )
        elif sym.name == "node_flag":
            return NodeFlag(
                flag_type=str(intermediate_repr(sym.arguments[0])),
                flag=str(intermediate_repr(sym.arguments[1])),
                flag_group=str(intermediate_repr(sym.arguments[2])),
                source=str(intermediate_repr(sym.arguments[3])),
            )
    except (RuntimeError, AttributeError): # Added AttributeError for safety
        # This happens when using clingo w/ CFFI and trying to access ".name" for symbols
        # that are not functions, or if sym is not a clingo Symbol object
        pass

    if clingo_cffi():
        # Clingo w/ CFFI will throw an exception on failure
        try:
            return sym.string # type: ignore[no-any-return]
        except RuntimeError:
            return str(sym)
    else:
        # It's assumed sym has 'string' attribute or can be converted to str
        return sym.string or str(sym) # type: ignore[no-any-return, union-attr]


def extract_args(model: List[Any], predicate_name: str) -> List[Tuple[Any, ...]]:
    """Extract the arguments to predicates with the provided name from a model.

    Pull out all the predicates with name ``predicate_name`` from the model, and
    return their intermediate representation.
    """
    # sym is assumed to be a clingo Symbol with 'arguments' and 'name'
    return [
        typing.cast(Tuple[Any, ...], intermediate_repr(sym.arguments))
        for sym in model
        if hasattr(sym, 'name') and sym.name == predicate_name # Ensure sym has name
    ]
