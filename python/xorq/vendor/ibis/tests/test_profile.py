import os
import pathlib

import pytest
import yaml

import xorq as xo
from xorq.vendor.ibis.backends import BaseBackend, Profile, Profiles


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
    assert default_profile_dir == pathlib.Path("~/.config/letsql/profiles").expanduser()
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
    profile.save()

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


def test_elide_secrets():
    os.environ["TEST_SECRET_USER"] = "test_user"
    os.environ["TEST_SECRET_PASSWORD"] = "very_secret_password"
    os.environ["SOME_ENV"] = "some_env"
    # expected
    expected_kwargs_dict_elided = dict(
        host="localhost-dummy",
        port=5432,
        user="${TEST_SECRET_USER}",
        password="${TEST_SECRET_PASSWORD}",
        database="test_db",
        secret="***elided***",
        nonsecret="visible_data",
        empty_param="",
        direct_env="${SOME_ENV}",
    )

    expected_kwargs_dict_raw = dict(
        host="localhost-dummy",
        port=5432,
        user="test_user",
        password="very_secret_password",
        database="test_db",
        secret="another_secret",
        nonsecret="visible_data",
        empty_param="",
        direct_env="${SOME_ENV}",
    )

    # create profile
    profile = Profile(
        con_name="postgres",
        kwargs_tuple=(
            ("host", "localhost-dummy"),
            ("port", 5432),
            ("user", "test_user"),
            ("password", "very_secret_password"),
            ("database", "test_db"),
            ("secret", "another_secret"),
            ("nonsecret", "visible_data"),
            ("empty_param", ""),
            ("direct_env", "${SOME_ENV}"),
        ),
    )

    # elide
    elided_profile = profile.elide_secrets()

    # check raw
    assert profile.kwargs_dict == expected_kwargs_dict_raw

    # no elision
    assert elided_profile.kwargs_dict == expected_kwargs_dict_elided

    # is it still yaml
    yaml_content = elided_profile.as_yaml()
    data = yaml.safe_load(yaml_content)

    # check that on disk is elided
    assert data["kwargs_dict"] == expected_kwargs_dict_elided
    # Make sure same back from disk
    profile.save()
    loaded_profile = profile.load(profile.hash_name)

    # no env for secret
    assert loaded_profile.kwargs_dict["secret"] == "***elided***"
    # assert profile == profile.load(profile.hash_name)

    # Clean up
    del os.environ["TEST_SECRET_USER"]
    del os.environ["TEST_SECRET_PASSWORD"]
    del os.environ["SOME_ENV"]


def test_missing_env_var_elided(monkeypatch, tmp_path):
    # NOTE: in the event a variable does not exist and it is a secret it will be elided
    # we will only save env vars that exist during save
    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)
    os.environ["EXISTING_ENV"] = "exists"

    profile = Profile(
        con_name="postgres",
        kwargs_tuple=(
            ("user", "${EXISTING_ENV}"),
            ("password", "${MISSING_ENV_VAR}"),  # This doesn't exist
            ("host", "localhost-dummy"),
            ("port", 5432),
        ),
    )

    profile.save()

    assert Profile.load(profile.hash_name).kwargs_dict["password"] == "***elided***"

    del os.environ["EXISTING_ENV"]


def test_missing_env_var_strict(monkeypatch, tmp_path):
    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)
    os.environ["EXISTING_ENV"] = "exists"
    expected_value_error = (
        r"Strict mode enabled, variables \$MISSING_ENV_VAR are not defined!"
    )
    profile = Profile(
        con_name="postgres",
        kwargs_tuple=(
            ("user", "${EXISTING_ENV}"),
            ("password", "BANANIBAL"),  # it will elide secrets
            ("host", "${MISSING_ENV_VAR}"),
            ("port", 5432),
        ),
    )

    profile_path = profile.save()
    assert profile_path.exists()

    with pytest.raises(ValueError, match=expected_value_error):
        Profile.load(profile.hash_name)

    del os.environ["EXISTING_ENV"]


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
    assert profile1.hash_name == profile2.hash_name
    # check sort
    assert profile1.kwargs_tuple == profile2.kwargs_tuple

    # check clone
    cloned = profile1.clone()
    assert cloned.hash_name == profile1.hash_name
