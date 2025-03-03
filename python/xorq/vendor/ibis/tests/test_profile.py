import os
import pathlib

import pytest

import xorq as xo
from xorq.vendor.ibis.backends import BaseBackend
from xorq.vendor.ibis.backends.profiles import Profile, Profiles, parse_env_vars


local_con_names = ("duckdb", "let", "datafusion", "pandas")
remote_connectors = (
    xo.postgres.connect_env,
    xo.postgres.connect_examples,
)
local_connectors = tuple(getattr(xo, con_name).connect for con_name in local_con_names)


@pytest.mark.parametrize(
    "con_name",
    local_con_names
    + (
        pytest.param(
            "invalid-con-name", marks=pytest.mark.xfail(reason="only valid con names")
        ),
    ),
)
def test_con_has_profile(con_name):
    con = getattr(xo, con_name).connect()
    assert isinstance(con, BaseBackend)
    profile = getattr(con, "_profile", None)
    assert isinstance(profile, Profile)
    assert profile.almost_equals(Profile.from_con(con))
    #
    other = profile.get_con()
    assert con.name == other.name
    # this doesn't work because _con_args, _con_kwargs doesn't get the defaults which are eventually invoked
    # assert hash(con) == hash(other)
    assert profile.almost_equals(other._profile)


@pytest.mark.parametrize("connect", remote_connectors)
def test_remote_con_works(connect):
    con = connect()
    assert isinstance(con, BaseBackend)
    profile = getattr(con, "_profile", None)
    assert isinstance(profile, Profile)
    assert profile.almost_equals(Profile.from_con(con))
    #
    other = profile.get_con()
    assert con.name == other.name
    # this doesn't work because _con_args, _con_kwargs doesn't get the defaults which are eventually invoked
    # assert hash(con) == hash(other)
    assert profile.almost_equals(other._profile)
    assert con.list_tables() == other.list_tables()


def test_profiles(monkeypatch, tmp_path):
    default_profile_dir = xo.options.profiles.profile_dir
    assert default_profile_dir == pathlib.Path("~/.config/xorq/profiles").expanduser()
    profiles = Profiles()
    assert profiles.profile_dir == default_profile_dir
    assert not profiles.list()  # why do this ?

    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)
    profiles = Profiles()
    assert profiles.profile_dir == tmp_path


@pytest.mark.parametrize("connector", remote_connectors + local_connectors)
def test_save_load(connector, monkeypatch, tmp_path):
    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)
    # In the is case letsql has a raw passwords so its value is
    # ***elided*** so we can't instantiate it
    os.environ["LETSQL_PASSWORD"] = "letsql"
    con = connector()
    profiles = Profiles()
    profile = con._profile
    profile.save(check_secrets=False)

    others = tuple(
        (
            profiles.get(profile.hash_name),
            profiles[profile.hash_name],
            profile.load(profile.hash_name),
        )
    )
    for other in others:
        assert profile == other
        assert con.list_tables() == other.get_con().list_tables()
    del os.environ["LETSQL_PASSWORD"]


def test_profile_hash_order_independence():
    # different order same kwargs
    profile1 = Profile(
        con_name="postgres",
        kwargs_tuple=(
            ("host", "localhost-dum"),
            ("port", 5432),
            ("user", "testuser"),
            ("database", "testdb"),
        ),
    )

    profile2 = Profile(
        con_name="postgres",
        kwargs_tuple=(
            ("port", 5432),
            ("database", "testdb"),
            ("user", "testuser"),
            ("host", "localhost-dum"),
        ),
    )

    # check hash order agnostic
    assert profile1.hash_name.split("_")[0] == profile2.hash_name.split("_")[0]
    assert profile1.hash_name.split("_")[1] != profile2.hash_name.split("_")[1]
    # check sort
    assert profile1.kwargs_tuple == profile2.kwargs_tuple

    # check clone
    cloned = profile1.clone()
    assert cloned.hash_name.split("_")[0] == profile1.hash_name.split("_")[0]


class TestParseEnvVars:
    def test_empty_dict(self):
        """Test with empty dictionary."""
        assert parse_env_vars({}) == {}

    def test_no_env_vars(self):
        """Test with dictionary containing no environment variables."""
        input_dict = {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "non_string": 123,
            "none_value": None,
            "empty_string": "",
        }
        assert parse_env_vars(input_dict) == input_dict

    def test_dollar_brace_format(self, monkeypatch):
        """Test with ${VAR} format environment variables."""
        # Set environment variables for testing
        monkeypatch.setenv("TEST_USER", "testuser")
        monkeypatch.setenv("TEST_PASSWORD", "secretpass")

        input_dict = {
            "host": "localhost",
            "port": 5432,
            "user": "${TEST_USER}",
            "password": "${TEST_PASSWORD}",
            "non_env": "regular_value",
        }

        expected = {
            "host": "localhost",
            "port": 5432,
            "user": "testuser",
            "password": "secretpass",
            "non_env": "regular_value",
        }

        assert parse_env_vars(input_dict) == expected

    def test_dollar_format(self, monkeypatch):
        """Test with $VAR format environment variables."""
        # Set environment variables for testing
        monkeypatch.setenv("TEST_USER", "testuser")
        monkeypatch.setenv("TEST_PASSWORD", "secretpass")

        input_dict = {
            "host": "localhost",
            "port": 5432,
            "user": "$TEST_USER",
            "password": "$TEST_PASSWORD",
            "non_env": "regular_value",
        }

        expected = {
            "host": "localhost",
            "port": 5432,
            "user": "testuser",
            "password": "secretpass",
            "non_env": "regular_value",
        }

        assert parse_env_vars(input_dict) == expected

    def test_mixed_formats(self, monkeypatch):
        """Test with mixed ${VAR} and $VAR formats."""
        # Set environment variables for testing
        monkeypatch.setenv("TEST_USER", "testuser")
        monkeypatch.setenv("TEST_PASSWORD", "secretpass")

        input_dict = {
            "host": "localhost",
            "port": 5432,
            "user": "${TEST_USER}",
            "password": "$TEST_PASSWORD",
            "non_env": "regular_value",
        }

        expected = {
            "host": "localhost",
            "port": 5432,
            "user": "testuser",
            "password": "secretpass",
            "non_env": "regular_value",
        }

        assert parse_env_vars(input_dict) == expected

    def test_non_string_values(self, monkeypatch):
        """Test with non-string values."""
        monkeypatch.setenv("TEST_VAR", "test_value")

        input_dict = {
            "string": "${TEST_VAR}",
            "integer": 123,
            "float": 45.67,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2},
        }

        expected = {
            "string": "test_value",
            "integer": 123,
            "float": 45.67,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2},
        }

        assert parse_env_vars(input_dict) == expected

    def test_missing_env_var(self, monkeypatch):
        """Test with missing environment variables - should raise ValueError."""
        monkeypatch.setenv("EXISTING_VAR", "exists")

        input_dict = {
            "existing": "${EXISTING_VAR}",
            "missing": "${MISSING_VAR}",
            "regular": "value",
        }

        with pytest.raises(ValueError) as exc_info:
            parse_env_vars(input_dict)

        assert "Error processing key 'missing': env var MISSING_VAR not found" in str(
            exc_info.value
        )

    def test_dollar_sign_in_string(self, monkeypatch):
        """Test with strings containing dollar signs but not as env vars."""
        input_dict = {
            "code": "a$b$c",
            "text": "This costs $5",
            "complex": "a${non-env}b",  # Not a proper env var syntax
        }

        expected = {
            "code": "a$b$c",
            "text": "This costs $5",
            "complex": "a${non-env}b",
        }

        assert parse_env_vars(input_dict) == expected

    def test_preserve_case(self, monkeypatch):
        """Test that environment variable case is preserved."""
        monkeypatch.setenv("UPPERCASE", "value1")
        monkeypatch.setenv("lowercase", "value2")
        monkeypatch.setenv("MixedCase", "value3")

        input_dict = {
            "var1": "${UPPERCASE}",
            "var2": "${lowercase}",
            "var3": "${MixedCase}",
        }

        expected = {
            "var1": "value1",
            "var2": "value2",
            "var3": "value3",
        }

        assert parse_env_vars(input_dict) == expected

    def test_nested_structures(self, monkeypatch):
        """Test how function handles nested structures (should only process top level)."""
        monkeypatch.setenv("TEST_VAR", "test_value")

        input_dict = {
            "top_level": "${TEST_VAR}",
            "nested_dict": {
                "env_var": "${TEST_VAR}",  # This should not be processed
                "normal": "value",
            },
            "list_with_vars": ["${TEST_VAR}", "normal"],  # This should not be processed
        }

        expected = {
            "top_level": "test_value",
            "nested_dict": {"env_var": "${TEST_VAR}", "normal": "value"},
            "list_with_vars": ["${TEST_VAR}", "normal"],
        }

        assert parse_env_vars(input_dict) == expected


def test_connection_with_env_vars_preserves_env_vars(monkeypatch, tmp_path):
    """Test that connections instantiated with env vars preserve them in profiles."""

    # Set up test environment
    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)

    # Set environment variables to match the existing profile values
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_PASSWORD", "postgres")
    monkeypatch.setenv("POSTGRES_DB", "ibis_testing")

    con_postgres = xo.postgres.connect(
        host="${POSTGRES_HOST}",
        user="${POSTGRES_USER}",
        password="${POSTGRES_PASSWORD}",
        database="${POSTGRES_DB}",
        port=5432,
    )

    # Get profile from connection
    profile = con_postgres._profile

    # Verify profile has env var references
    assert profile.kwargs_dict["host"] == "${POSTGRES_HOST}"
    assert profile.kwargs_dict["user"] == "${POSTGRES_USER}"
    assert profile.kwargs_dict["password"] == "${POSTGRES_PASSWORD}"
    assert profile.kwargs_dict["database"] == "${POSTGRES_DB}"

    # Save profile
    profile.save(alias="pg_env_var_test", check_secrets=False)

    # Create Profiles instance to load profiles
    profiles = Profiles(profile_dir=tmp_path)

    # Get profiles in different ways
    loaded_profiles = [
        profiles.get(profile.hash_name),
        profiles[profile.hash_name],
        profile.load(profile.hash_name, profile_dir=tmp_path),
    ]

    # Verify all loaded profiles have env var references
    for loaded_profile in loaded_profiles:
        assert loaded_profile.kwargs_dict["host"] == "${POSTGRES_HOST}"
        assert loaded_profile.kwargs_dict["user"] == "${POSTGRES_USER}"
        assert loaded_profile.kwargs_dict["password"] == "${POSTGRES_PASSWORD}"

        # Create connection from loaded profile
        loaded_con = loaded_profile.get_con()

        # Verify the connection's profile still has env var references
        assert loaded_con._profile is not None
        assert loaded_con._profile.kwargs_dict["host"] == "${POSTGRES_HOST}"
        assert loaded_con._profile.kwargs_dict["user"] == "${POSTGRES_USER}"
        assert loaded_con._profile.kwargs_dict["password"] == "${POSTGRES_PASSWORD}"

        # Test that the connection works by comparing to a simple list_tables call
        tables1 = con_postgres.list_tables()
        tables2 = loaded_con.list_tables()
        assert tables1 == tables2


class TestCheckForExposedSecrets:
    def test_password_no_env_var(self):
        """Test that a profile with password not using env var is rejected."""
        profile = Profile(
            con_name="postgres",
            kwargs_tuple=(
                ("host", "localhost"),
                ("port", 5432),
                ("database", "postgres"),
                ("user", "postgres"),
                ("password", "secret"),  # Not using env var
            ),
        )

        with pytest.raises(ValueError) as excinfo:
            profile._check_for_exposed_secrets(check_secrets=True)

        # Check error message contains password
        assert "'password'" in str(excinfo.value)
        assert "$password or ${password}" in str(excinfo.value)

    def test_password_with_env_var_dollar(self):
        """Test that a profile with password using $ENV_VAR format is accepted."""
        profile = Profile(
            con_name="postgres",
            kwargs_tuple=(
                ("host", "localhost"),
                ("port", 5432),
                ("database", "postgres"),
                ("user", "postgres"),
                ("password", "$PASSWORD"),  # Using env var
            ),
        )

        # Should not raise an error
        profile._check_for_exposed_secrets(check_secrets=True)

    def test_password_with_env_var_dollar_brace(self):
        """Test that a profile with password using ${ENV_VAR} format is accepted."""
        profile = Profile(
            con_name="postgres",
            kwargs_tuple=(
                ("host", "localhost"),
                ("port", 5432),
                ("database", "postgres"),
                ("user", "postgres"),
                ("password", "${PASSWORD}"),  # Using env var
            ),
        )

        # Should not raise an error
        profile._check_for_exposed_secrets(check_secrets=True)

    def test_postgres_specific_secret_keys(self):
        """Test that postgres-specific secret keys are checked."""
        profile = Profile(
            con_name="postgres",
            kwargs_tuple=(
                ("host", "localhost"),
                ("port", 5432),
                ("database", "postgres"),
                ("user", "postgres"),
                ("password", "$PASSWORD"),  # Using env var
                ("sslcert", "/path/to/cert"),  # Not using env var
            ),
        )

        with pytest.raises(ValueError) as excinfo:
            profile._check_for_exposed_secrets(check_secrets=True)

        # Check error message contains sslcert
        assert "'sslcert'" in str(excinfo.value)

    def test_snowflake_specific_secret_keys(self):
        """Test that snowflake-specific secret keys are checked."""
        profile = Profile(
            con_name="snowflake",
            kwargs_tuple=(
                ("host", "localhost"),
                ("database", "snowflake"),
                ("password", "$PASSWORD"),  # Using env var
                (
                    "user",
                    "snowuser",
                ),  # Not using env var - snowflake treats this as sensitive
            ),
        )

        with pytest.raises(ValueError) as excinfo:
            profile._check_for_exposed_secrets(check_secrets=True)

        # Check error message contains user
        assert "'user'" in str(excinfo.value)

    def test_check_secrets_disabled(self):
        """Test that check_secrets=False allows profiles with secrets."""
        profile = Profile(
            con_name="postgres",
            kwargs_tuple=(
                ("host", "localhost"),
                ("port", 5432),
                ("database", "postgres"),
                ("user", "postgres"),
                ("password", "secret"),  # Not using env var
            ),
        )

        # Should not raise an error when check_secrets=False
        profile._check_for_exposed_secrets(check_secrets=False)

    def test_multiple_exposed_secrets(self):
        """Test error message when multiple secrets are exposed."""
        profile = Profile(
            con_name="snowflake",
            kwargs_tuple=(
                ("host", "localhost"),
                ("database", "snowflake"),
                ("password", "secret"),  # Not using env var
                ("user", "admin"),  # Not using env var
                ("account", "acc123"),  # Not using env var
            ),
        )

        with pytest.raises(ValueError) as excinfo:
            profile._check_for_exposed_secrets(check_secrets=True)

        # Check error message contains all secrets
        error_msg = str(excinfo.value)
        assert "'password'" in error_msg
        assert "'user'" in error_msg
        assert "'account'" in error_msg

    def test_unknown_backend_defaults_to_password(self):
        """Test that unknown backends default to checking password."""
        profile = Profile(
            con_name="duckdb",  # Not in the secret_keys dict
            kwargs_tuple=(
                ("path", "mydb.duckdb"),
                ("password", "secret"),  # Not using env var
            ),
        )

        with pytest.raises(ValueError) as excinfo:
            profile._check_for_exposed_secrets(check_secrets=True)

        # Check error message contains password
        assert "'password'" in str(excinfo.value)

    def test_save_method_calls_check_secrets_fail_then_pass(
        self, tmp_path, monkeypatch
    ):
        """Test that save() method calls _check_for_exposed_secrets."""
        profile = Profile(
            con_name="postgres",
            kwargs_tuple=(
                ("host", "localhost"),
                ("port", 5432),
                ("database", "postgres"),
                ("user", "postgres"),
                ("password", "secret"),  # Not using env var
            ),
        )

        # Override the profile directory for testing
        monkeypatch.setattr("xorq.options.profiles.profile_dir", tmp_path)

        with pytest.raises(ValueError) as excinfo:
            profile.save()

        assert "'password'" in str(excinfo.value)

        # Should succeed with check_secrets=False
        profile.save(check_secrets=False)
