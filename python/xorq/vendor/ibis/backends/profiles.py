import itertools
import json
import os
import re
from pathlib import Path

import dask
import toolz
import yaml
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
)

import xorq as xo
from xorq.common.utils.inspect_utils import get_arguments


compiled_env_var_re = re.compile("^(?:\${(.*)}$)|(?:\$(.*))$")


@frozen
class Profiles:
    profile_dir = field(validator=optional(instance_of(Path)), default=None)

    def __attrs_post_init__(self):
        if self.profile_dir is None:
            object.__setattr__(self, "profile_dir", xo.options.profiles.profile_dir)
        if not self.profile_dir.exists():
            self.profile_dir.mkdir(exist_ok=True, parents=True)

    def get(self, name):
        return Profile.load(name, profile_dir=self.profile_dir)

    def __getattr__(self, stem):
        try:
            return self.get(
                next(el.name for el in self.profile_dir.iterdir() if el.stem == stem)
            )
        except Exception:
            return object.__getattribute__(self, stem)

    def __getitem__(self, stem):
        return self.get(
            next(el.name for el in self.profile_dir.iterdir() if el.stem == stem)
        )

    def __dir__(self):
        return tuple(el for el in self.list() if el.isidentifier())

    def list(self):
        return tuple(el.stem for el in self.profile_dir.iterdir())

    def _ipython_key_completions_(self):
        return self.list()


@frozen
class Profile:
    con_name = field(validator=instance_of(str))
    kwargs_tuple = field(validator=instance_of(tuple))
    idx = field(validator=instance_of(int), factory=itertools.count().__next__)

    @con_name.validator
    def validate_con_name(self, attr, value):
        assert next(
            (ep for ep in xo._load_entry_points() if ep.name == value),
            None,
        )

    def __attrs_post_init__(self):
        # Sort kwargs_tuple after initialization
        object.__setattr__(self, "kwargs_tuple", tuple(sorted(self.kwargs_tuple)))

    @property
    def kwargs_dict(self):
        return {k: v for k, v in self.kwargs_tuple}

    @property
    def hash_name(self):
        dask_hash = dask.base.tokenize(
            json.dumps(
                toolz.dissoc(self.as_dict(), "idx"),
            )
        )
        return f"{dask_hash}_{self.idx}"

    def get_con(self, **kwargs):
        """Create a connection using this profile's parameters."""
        _kwargs = dict(self.kwargs_tuple) | kwargs
        connect = getattr(xo.load_backend(self.con_name), "connect")
        con = connect(**_kwargs)
        return con

    def clone(self, idx=None, **kwargs):
        idx = idx if idx is not None else self.idx
        kwargs_tuple = tuple(
            {
                **dict(self.kwargs_tuple),
                **kwargs,
            }.items()
        )
        return type(self)(
            con_name=self.con_name,
            kwargs_tuple=kwargs_tuple,
            idx=idx,
        )

    def as_dict(self):
        return {
            name: getattr(self, name)
            for name in (attr.name for attr in self.__attrs_attrs__)
        }

    def as_json(self):
        return json.dumps(self.as_dict())

    def as_yaml(self):
        return yaml.safe_dump(
            dict(con_name=self.con_name, kwargs_dict=self.kwargs_dict, idx=self.idx)
        )

    def _check_for_exposed_secrets(self, check_secrets: bool) -> None:
        """Check if profile contains exposed secret keys.

        Raises
        ------
        ValueError
            If profile contains exposed secret keys not using environment variables
        """

        if check_secrets:
            # Define secret keys by database type
            # TODO: Add more database types as needed
            # maybe user sets this in options
            secret_keys = {
                "postgres": [
                    "password",
                    "sslcert",
                    "sslkey",
                    "sslrootcert",
                    "sslcrl",
                    "options",
                    "passfile",
                ],
                "snowflake": [
                    "password",
                    "user",
                    "account",
                    "token",
                    "private_key",
                    "private_key_path",
                    "oauth_token",
                ],
                # Add more database types as needed
            }

            # default to just password
            relevant_secrets = secret_keys.get(
                self.con_name, ["password"]
            )  # Default to just password

            exposed_secrets = [
                key
                for key, value in self.kwargs_dict.items()
                if key in relevant_secrets
                and not (isinstance(value, str) and compiled_env_var_re.match(value))
            ]

            if exposed_secrets:
                secrets_list = ", ".join(f"'{key}'" for key in exposed_secrets)
                env_var_examples = ", ".join(
                    f"${key} or ${{{key}}}" for key in exposed_secrets
                )
                raise ValueError(
                    f"Profile contains exposed secret keys: {secrets_list}. "
                    f"Use environment variables ({env_var_examples}) for these values."
                )

    def save(self, profile_dir=None, alias=None, clobber=False, check_secrets=True):
        self._check_for_exposed_secrets(check_secrets)
        path = self.get_path(self.hash_name, profile_dir=profile_dir)
        if not path.exists():
            path.write_text(self.as_yaml())
        if alias:
            alias_path = self.get_path(alias, profile_dir=profile_dir)
            if alias_path.exists():
                if not clobber:
                    raise ValueError
                alias_path.unlink()
            alias_path.symlink_to(path)
            return alias_path
        return path

    def almost_equals(self, other):
        return self.clone(idx=-1) == other.clone(idx=-1)

    @classmethod
    def get_path(cls, name, profile_dir=None):
        profile_dir = profile_dir or xo.options.profiles.profile_dir
        profile_dir.mkdir(exist_ok=True, parents=True)
        path = profile_dir.joinpath(name).with_suffix(".yaml")
        return path

    @classmethod
    def load(cls, name, profile_dir=None):
        path = cls.get_path(name, profile_dir=profile_dir)
        env = yaml.safe_load(path.read_text())
        con_name = env.get("con_name")
        kwargs_dict = env.get("kwargs_dict")
        idx = env.get("idx")
        sorted_kwargs = tuple(sorted(kwargs_dict.items()))
        return cls(con_name=con_name, kwargs_tuple=sorted_kwargs, idx=idx)

    @classmethod
    def from_con(cls, con, *args, **kwargs):
        """Create a Profile from a connection, preserving env var references if possible."""

        def get_combined_arguments():
            # these are the env-mapped values
            arguments0 = get_arguments(
                con.do_connect, *con._con_args, **con._con_kwargs
            )
            # these are the "raw" values (if passed)
            arguments1 = toolz.valfilter(
                bool, get_arguments(con.do_connect, *args, **kwargs)
            )
            assert not arguments0.get("args")
            assert not arguments1.get("args")
            arguments = toolz.dissoc(arguments0 | arguments1, "args")
            return arguments

        if con.name == "xorq_flight":
            return None

        kwargs_name = "config" if con.name == "duckdb" else "kwargs"
        arguments = get_combined_arguments()
        kwargs = toolz.dissoc(arguments, kwargs_name) | arguments.get(kwargs_name, {})

        # Fix port type if needed
        if (
            "port" in kwargs
            and kwargs["port"] is not None
            and isinstance(kwargs["port"], str)
        ):
            kwargs["port"] = int(kwargs["port"])

        return cls(con_name=con.name, kwargs_tuple=tuple(sorted(kwargs.items())))


def maybe_process_env_var(obj):
    if isinstance(obj, str) and (match := compiled_env_var_re.match(obj)):
        # this will match on "$"/"${}" and then raise on env_value is None
        env_var = next(filter(None, match.groups()), None)
        env_value = os.environ.get(env_var)
        if env_value is None:
            raise ValueError(f"env var {env_var} not found")
        else:
            return env_value
    else:
        return obj


def parse_env_vars(kwargs_dict: dict) -> dict:
    """Process all environment variables in a dictionary.

    Uses maybe_process_env_var internally to ensure consistent behavior.
    """
    processed_kwargs = {}

    # get env keys
    env_var_keys = [
        k
        for k, v in kwargs_dict.items()
        if isinstance(v, str) and compiled_env_var_re.match(v)
    ]

    # possibly parse
    for k, v in kwargs_dict.items():
        if k in env_var_keys:
            try:
                processed_kwargs[k] = maybe_process_env_var(v)
            except ValueError as e:
                # Re-raise with more context if needed
                raise ValueError(f"Error processing key '{k}': {str(e)}")
        else:
            processed_kwargs[k] = v

    return processed_kwargs
