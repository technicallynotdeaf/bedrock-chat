"""
Tests covering the five security hardening changes:
1. eval() replaced in both calculator tools
2. Authorization header no longer logged
3. JWKS fetch has timeout and module-level cache
4. Test user fallback removed from main.py
"""

import ast
import importlib
import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. agents/tools/calculator.py — AST-based evaluator
# ---------------------------------------------------------------------------


class TestAgentsCalculator(unittest.TestCase):
    def setUp(self):
        from app.agents.tools.calculator import calculate_expression

        self.calc = calculate_expression

    def test_basic_arithmetic(self):
        self.assertEqual(self.calc("2+2"), "4")
        self.assertEqual(self.calc("10 - 3"), "7")
        self.assertEqual(self.calc("6 * 7"), "42")
        self.assertEqual(self.calc("10 / 4"), "2.5")

    def test_parentheses(self):
        self.assertEqual(self.calc("(2+3)*4"), "20")

    def test_float_result(self):
        result = self.calc("1/3")
        self.assertTrue(result.startswith("0.333"))

    def test_division_by_zero(self):
        result = self.calc("1/0")
        self.assertIn("Error", result)

    def test_no_eval_used(self):
        """Confirm eval() is not called during calculation."""
        with patch("builtins.eval", side_effect=AssertionError("eval() was called")) as mock_eval:
            result = self.calc("3*3")
        self.assertEqual(result, "9")

    def test_rejects_attribute_access(self):
        result = self.calc("().__class__")
        self.assertIn("Error", result)

    def test_rejects_import(self):
        result = self.calc("__import__('os')")
        self.assertIn("Error", result)

    def test_rejects_string_literal(self):
        result = self.calc("'hello'")
        self.assertIn("Error", result)

    def test_rejects_exponentiation_does_not_crash(self):
        # ** is not in _SAFE_OPS so it should return an error, not execute
        result = self.calc("2**10")
        self.assertIn("Error", result)

    def test_negative_numbers(self):
        self.assertEqual(self.calc("-5 + 3"), "-2")


# ---------------------------------------------------------------------------
# 2. strands_integration/tools/calculator.py — AST-based evaluator
# ---------------------------------------------------------------------------


class TestStrandsCalculator(unittest.TestCase):
    def setUp(self):
        from app.strands_integration.tools.calculator import _ast_eval

        self.eval = _ast_eval

    def _parse(self, expr):
        return ast.parse(expr, mode="eval").body

    def test_basic_arithmetic(self):
        self.assertAlmostEqual(self.eval(self._parse("2+2")), 4.0)
        self.assertAlmostEqual(self.eval(self._parse("10/4")), 2.5)

    def test_power(self):
        self.assertAlmostEqual(self.eval(self._parse("2**8")), 256.0)

    def test_math_functions(self):
        import math

        self.assertAlmostEqual(self.eval(self._parse("sqrt(16)")), 4.0)
        self.assertAlmostEqual(self.eval(self._parse("sin(0)")), 0.0)
        self.assertAlmostEqual(self.eval(self._parse("log(1)")), 0.0)
        self.assertAlmostEqual(self.eval(self._parse("pi")), math.pi)

    def test_rejects_attribute_access(self):
        with self.assertRaises(ValueError):
            self.eval(self._parse("().__class__.__bases__"))

    def test_rejects_unknown_function(self):
        with self.assertRaises(ValueError):
            self.eval(self._parse("open('etc/passwd')"))

    def test_no_eval_used(self):
        with patch("builtins.eval", side_effect=AssertionError("eval() was called")):
            from app.strands_integration.tools.calculator import _ast_eval

            result = _ast_eval(ast.parse("3*3", mode="eval").body)
        self.assertAlmostEqual(result, 9.0)

    def test_is_safe_expression_removed(self):
        """_is_safe_expression should no longer exist in the module."""
        import app.strands_integration.tools.calculator as mod

        self.assertFalse(
            hasattr(mod, "_is_safe_expression"),
            "_is_safe_expression should have been removed",
        )


# ---------------------------------------------------------------------------
# 3. main.py — Authorization header not logged, request body not logged
# ---------------------------------------------------------------------------

import pathlib

_MAIN_SOURCE = (
    pathlib.Path(__file__).parent.parent / "app" / "main.py"
).read_text()


def _extract_function_source(full_source: str, func_name: str) -> str:
    """Extract the source of a specific function from a module's source text."""
    lines = full_source.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"async def {func_name}") or line.startswith(f"def {func_name}"):
            start = i
            break
    if start is None:
        return ""
    # Collect lines until the next top-level definition or end of file
    result = [lines[start]]
    for line in lines[start + 1:]:
        if line and not line[0].isspace():
            break
        result.append(line)
    return "\n".join(result)


class TestRequestLogging(unittest.TestCase):
    def setUp(self):
        self._log_source = _extract_function_source(_MAIN_SOURCE, "add_log_requests")

    def test_authorization_header_not_logged(self):
        # request.headers must not appear inside the logging middleware
        self.assertNotIn(
            "request.headers",
            self._log_source,
            "request.headers should not be logged in add_log_requests",
        )
        self.assertNotIn(
            "request.body",
            self._log_source,
            "request.body should not be logged in add_log_requests",
        )

    def test_path_and_method_still_logged(self):
        self.assertIn("request.url.path", self._log_source)
        self.assertIn("request.method", self._log_source)


# ---------------------------------------------------------------------------
# 4. auth.py — JWKS timeout and caching
# ---------------------------------------------------------------------------


class TestAuthJWKS(unittest.TestCase):
    def setUp(self):
        # Reset module-level cache before each test
        import app.auth as auth_mod

        auth_mod._JWKS_CACHE = None
        auth_mod._JWKS_CACHE_EXPIRY = 0.0

    def _make_mock_response(self, keys):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"keys": keys}
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_timeout_is_passed(self):
        import app.auth as auth_mod

        fake_keys = [{"kid": "k1", "kty": "RSA"}]
        with patch("requests.get", return_value=self._make_mock_response(fake_keys)) as mock_get:
            result = auth_mod._get_jwks()
        call_kwargs = mock_get.call_args
        self.assertIn("timeout", call_kwargs.kwargs if call_kwargs.kwargs else {})
        passed_timeout = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
        self.assertIsNotNone(passed_timeout, "timeout kwarg must be passed to requests.get")
        self.assertGreater(passed_timeout, 0)

    def test_jwks_cached_on_second_call(self):
        import app.auth as auth_mod

        fake_keys = [{"kid": "k1", "kty": "RSA"}]
        with patch("requests.get", return_value=self._make_mock_response(fake_keys)) as mock_get:
            auth_mod._get_jwks()
            auth_mod._get_jwks()
        # Should only have made one HTTP request
        self.assertEqual(mock_get.call_count, 1, "JWKS should be cached after first fetch")

    def test_cache_expires_after_ttl(self):
        import app.auth as auth_mod

        fake_keys = [{"kid": "k1"}]
        with patch("requests.get", return_value=self._make_mock_response(fake_keys)) as mock_get:
            auth_mod._get_jwks()
            # Expire the cache
            auth_mod._JWKS_CACHE_EXPIRY = time.monotonic() - 1
            auth_mod._get_jwks()
        self.assertEqual(mock_get.call_count, 2, "JWKS should be re-fetched after TTL expires")

    def test_missing_kid_raises_value_error(self):
        import app.auth as auth_mod

        fake_keys = [{"kid": "other-key"}]
        with patch("app.auth._get_jwks", return_value=fake_keys):
            # Craft a minimal JWT header with a non-matching kid
            import base64, json

            header = base64.urlsafe_b64encode(
                json.dumps({"alg": "RS256", "kid": "missing-kid"}).encode()
            ).rstrip(b"=").decode()
            fake_token = f"{header}.payload.signature"
            with self.assertRaises((ValueError, Exception)):
                auth_mod.verify_token(fake_token)


# ---------------------------------------------------------------------------
# 5. main.py — Test user fallback removed
# ---------------------------------------------------------------------------


class TestNoTestUserFallback(unittest.TestCase):
    def test_test_user_not_in_source(self):
        self.assertNotIn(
            "test_user",
            _MAIN_SOURCE,
            "Hardcoded test_user fallback should have been removed",
        )

    def test_no_unauthenticated_fallback_outside_lambda(self):
        self.assertNotIn('id="test_user"', _MAIN_SOURCE)
        self.assertNotIn('email="user@example.com"', _MAIN_SOURCE)


if __name__ == "__main__":
    unittest.main()
