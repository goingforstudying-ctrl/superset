# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Unit tests for user-related MCP schemas."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from sqlalchemy.orm.exc import DetachedInstanceError

from superset.mcp_service.user.schemas import (
    sanitize_for_llm_context,
    serialize_user_object,
    UserInfo,
)


def test_user_info_rejects_bare_string_for_roles() -> None:
    """A plain string must not be silently split into individual characters."""
    with pytest.raises(ValidationError):
        UserInfo(roles="Admin")


def test_user_info_preserves_empty_roles_list() -> None:
    """Empty roles should remain [] so callers can distinguish it from None."""
    info = UserInfo(roles=[])
    assert info.roles == []


def test_user_info_coerces_role_objects_to_names() -> None:
    """Role-like ORM objects must be converted to their .name strings."""
    role_admin = MagicMock()
    role_admin.name = "Admin"
    role_alpha = MagicMock()
    role_alpha.name = "Alpha"

    info = UserInfo(roles=[role_admin, role_alpha])

    assert info.roles == ["Admin", "Alpha"]


def test_user_info_ignores_role_with_detached_instance() -> None:
    """Detached ORM roles must not blow up serialization."""
    role_good = MagicMock()
    role_good.name = "Admin"

    class DetachedRole:
        """Mock ORM role that raises DetachedInstanceError on name access."""

        @property
        def name(self) -> str:
            raise DetachedInstanceError()

    role_detached = DetachedRole()

    info = UserInfo(roles=[role_good, role_detached])

    assert info.roles == ["Admin"]


def test_serialize_user_object_round_trip_with_empty_roles() -> None:
    """serialize_user_object must produce UserInfo.roles == [] for empty roles."""
    user = MagicMock()
    user.id = 1
    user.username = "admin"
    user.first_name = "Admin"
    user.last_name = "User"
    user.active = True
    user.email = "admin@example.com"
    user.changed_on = None
    user.roles = []

    info = serialize_user_object(user, include_sensitive=True, include_roles=True)

    assert info is not None
    assert info.roles == []
    assert info.username == "admin"
    assert info.first_name == sanitize_for_llm_context(
        "Admin", field_path=("first_name",)
    )
    assert info.last_name == sanitize_for_llm_context(
        "User", field_path=("last_name",)
    )
    assert info.active is True
    assert info.email == "admin@example.com"


def test_serialize_user_object_round_trip_with_role_objects() -> None:
    """Full from_attributes path through serialize_user_object -> UserInfo."""
    role_admin = MagicMock()
    role_admin.name = "Admin"

    user = MagicMock()
    user.id = 1
    user.username = "admin"
    user.first_name = "Admin"
    user.last_name = "User"
    user.active = True
    user.email = "admin@example.com"
    user.changed_on = None
    user.roles = [role_admin]

    info = serialize_user_object(user, include_sensitive=True, include_roles=True)

    assert info is not None
    assert info.roles == ["Admin"]
    assert info.username == "admin"
    assert info.first_name == sanitize_for_llm_context(
        "Admin", field_path=("first_name",)
    )
    assert info.last_name == sanitize_for_llm_context(
        "User", field_path=("last_name",)
    )
    assert info.active is True
    assert info.email == "admin@example.com"


def test_serialize_user_object_skips_roles_when_include_roles_false() -> None:
    """serialize_user_object must return roles=None when include_roles=False."""
    role_admin = MagicMock()
    role_admin.name = "Admin"

    user = MagicMock()
    user.id = 1
    user.username = "admin"
    user.first_name = "Admin"
    user.last_name = "User"
    user.active = True
    user.email = "admin@example.com"
    user.changed_on = None
    user.roles = [role_admin]

    info = serialize_user_object(user, include_sensitive=True, include_roles=False)

    assert info is not None
    assert info.roles is None
    assert info.email == "admin@example.com"


def test_serialize_user_object_skips_email_when_include_sensitive_false() -> None:
    """serialize_user_object must return email=None when include_sensitive=False."""
    role_admin = MagicMock()
    role_admin.name = "Admin"

    user = MagicMock()
    user.id = 1
    user.username = "admin"
    user.first_name = "Admin"
    user.last_name = "User"
    user.active = True
    user.email = "admin@example.com"
    user.changed_on = None
    user.roles = [role_admin]

    info = serialize_user_object(user, include_sensitive=False, include_roles=True)

    assert info is not None
    assert info.email is None
    assert info.roles is None


def test_serialize_user_object_returns_none_for_falsy_user() -> None:
    """serialize_user_object must return None for user=None, user=0, user=''."""
    assert serialize_user_object(None) is None
    assert serialize_user_object(0) is None
    assert serialize_user_object("") is None
