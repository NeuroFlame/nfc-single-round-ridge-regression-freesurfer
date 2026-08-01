"""Regression tests for ridge CSV header compatibility."""

import logging
import os
import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd
from computation.input_validation import validate_and_get_inputs
from computation.inputs import load_inputs


class InputHeaderCompatibilityTests(unittest.TestCase):
    """Verify legacy case-insensitive and whitespace-tolerant header matching."""

    def setUp(self):
        """Create standard computation parameters and a test logger."""
        self.parameters = {
            "Covariates": {"age": "float"},
            "Dependents": {"ROI": "float"},
        }
        self.logger = Mock(spec=logging.Logger)

    def _validate(self, covariate_header: str, dependent_header: str):
        """Write one-row CSV inputs and return their validation result."""
        with tempfile.TemporaryDirectory() as temp_dir:
            covariates_path = os.path.join(temp_dir, "covariates.csv")
            data_path = os.path.join(temp_dir, "data.csv")
            pd.DataFrame({covariate_header: [42]}).to_csv(covariates_path, index=False)
            pd.DataFrame({dependent_header: [3.5]}).to_csv(data_path, index=False)
            return validate_and_get_inputs(
                covariates_path,
                data_path,
                self.parameters,
                self.logger,
            )

    def test_headers_are_matched_case_insensitively(self):
        """Accept uppercase inputs and restore configured column names."""
        is_valid, covariates, dependents = self._validate("AGE", "roi")

        self.assertTrue(is_valid)
        self.assertEqual(["age"], covariates.columns.tolist())
        self.assertEqual(["ROI"], dependents.columns.tolist())

    def test_headers_ignore_surrounding_whitespace(self):
        """Accept surrounding header whitespace and restore configured names."""
        is_valid, covariates, dependents = self._validate(" age ", " ROI ")

        self.assertTrue(is_valid)
        self.assertEqual(["age"], covariates.columns.tolist())
        self.assertEqual(["ROI"], dependents.columns.tolist())

    def test_missing_header_fails_validation(self):
        """Reject an input that lacks a configured header."""
        is_valid, covariates, dependents = self._validate("height", "ROI")

        self.assertFalse(is_valid)
        self.assertIsNone(covariates)
        self.assertIsNone(dependents)
        self.assertTrue(
            any(
                "headers do not contain all expected headers" in str(call)
                for call in self.logger.error.call_args_list
            )
        )

    def test_duplicate_normalized_headers_fail_validation(self):
        """Reject distinct CSV headers that normalize to the same name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            covariates_path = os.path.join(temp_dir, "covariates.csv")
            data_path = os.path.join(temp_dir, "data.csv")
            pd.DataFrame([[42, 43]], columns=["age", " AGE "]).to_csv(
                covariates_path, index=False
            )
            pd.DataFrame({"ROI": [3.5]}).to_csv(data_path, index=False)

            is_valid, covariates, dependents = validate_and_get_inputs(
                covariates_path,
                data_path,
                self.parameters,
                self.logger,
            )

        self.assertFalse(is_valid)
        self.assertIsNone(covariates)
        self.assertIsNone(dependents)
        self.assertTrue(
            any(
                "duplicate headers after normalization" in str(call)
                for call in self.logger.error.call_args_list
            )
        )

    def test_missing_required_parameter_names_the_section(self):
        """Expose a useful terminal error when run parameters are incomplete."""
        with self.assertRaisesRegex(
            ValueError,
            "Missing required computation parameter 'Dependents'",
        ):
            load_inputs(
                "unused-covariates.csv",
                "unused-data.csv",
                {"Covariates": {"age": "float"}},
                self.logger,
            )


if __name__ == "__main__":
    unittest.main()
