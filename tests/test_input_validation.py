"""Regression tests for ridge CSV header compatibility."""

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
from computation.input_validation import InputValidationError, validate_and_get_inputs
from computation.inputs import load_inputs
from framework.errors import record_terminal_error


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
        with self.assertRaisesRegex(
            InputValidationError, "Missing required covariate column: age"
        ):
            self._validate("height", "ROI")

    def test_duplicate_normalized_headers_fail_validation(self):
        """Reject distinct CSV headers that normalize to the same name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            covariates_path = os.path.join(temp_dir, "covariates.csv")
            data_path = os.path.join(temp_dir, "data.csv")
            pd.DataFrame([[42, 43]], columns=["age", " AGE "]).to_csv(
                covariates_path, index=False
            )
            pd.DataFrame({"ROI": [3.5]}).to_csv(data_path, index=False)

            with self.assertRaisesRegex(InputValidationError, "duplicate column names"):
                validate_and_get_inputs(
                    covariates_path,
                    data_path,
                    self.parameters,
                    self.logger,
                )

    def test_data_errors_reach_terminal_report_without_local_row_details(self):
        cases = [
            (
                {"age": [42]},
                {"private_unused_header": [3.5]},
                {},
                "Missing required dependent column: ROI",
            ),
            ({"age": [42]}, {"ROI": [3.5, 4.5]}, {}, "different row counts"),
            (
                {"age": ["private_invalid_cell"]},
                {"ROI": [3.5]},
                {},
                "Invalid or missing values detected (affected rows: 1)",
            ),
            (
                {"age": ["private_invalid_cell"]},
                {"ROI": [3.5]},
                {"Covariates": {"age": "int"}},
                "Could not validate configured column age as int",
            ),
            (
                {"age": [42]},
                {"ROI": [3.5]},
                {"Covariates": {"age": "unsupported"}},
                "Allowed datatypes",
            ),
        ]
        for covariates, data, overrides, expected in cases:
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                covariates_path = os.path.join(directory, "covariates.csv")
                data_path = os.path.join(directory, "data.csv")
                pd.DataFrame(covariates).to_csv(covariates_path, index=False)
                pd.DataFrame(data).to_csv(data_path, index=False)
                try:
                    load_inputs(
                        covariates_path,
                        data_path,
                        {**self.parameters, **overrides},
                        self.logger,
                    )
                except InputValidationError as error:
                    self.assertIn(expected, str(error))
                    record_terminal_error(
                        directory,
                        "fit_local_models",
                        error,
                        origin="site",
                        stage="task_execution",
                    )
                else:
                    self.fail("Expected an actionable input validation error")
                marker = json.loads(
                    Path(directory, ".neuroflame_error.json").read_text()
                )
                self.assertIn(expected, marker["message"])
                for private in [
                    "private_unused_header",
                    "private_invalid_cell",
                    covariates_path,
                    data_path,
                ]:
                    self.assertNotIn(private, marker["message"] + marker["traceback"])

    def test_unexpected_file_errors_keep_local_paths_out_of_report(self):
        with self.assertRaisesRegex(ValueError, "Invalid run input") as caught:
            load_inputs(
                "private_missing_covariates.csv",
                "private_data.csv",
                self.parameters,
                self.logger,
            )
        self.assertNotIn("private_", str(caught.exception))

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
