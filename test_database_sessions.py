import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from database import close_session_quietly
from services.auth import get_current_user


class DatabaseSessionTests(unittest.TestCase):
    def test_cleanup_swallows_disconnect_and_invalidates_session(self):
        db = MagicMock()
        db.close.side_effect = RuntimeError("SSL connection closed")

        close_session_quietly(db)

        db.close.assert_called_once_with()
        db.invalidate.assert_called_once_with()

    @patch("services.auth.decode_access_token", return_value={"sub": "7"})
    @patch("services.auth.SessionLocal")
    def test_authentication_releases_its_session_before_returning(
        self,
        session_factory,
        _decode_token,
    ):
        user = SimpleNamespace(id=7, name="Alex", role="admin", is_active=True)
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = user
        session_factory.return_value = db

        result = get_current_user(token="valid-token")

        self.assertIs(result, user)
        db.expunge.assert_called_once_with(user)
        db.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
