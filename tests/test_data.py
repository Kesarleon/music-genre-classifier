import unittest
import os
import pandas as pd
import sys

# Add src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generate_data import generate_data

class TestDataGeneration(unittest.TestCase):

    def setUp(self):
        """Set up for the tests."""
        self.output_dir = "tests/temp_data"
        self.output_path = os.path.join(self.output_dir, "raw_music_data.csv")
        # Ensure the temp directory is clean before each test
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, f))
            os.rmdir(self.output_dir)

    def tearDown(self):
        """Tear down after the tests."""
        # Clean up the created files and directory
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        if os.path.exists(self.output_dir):
            os.rmdir(self.output_dir)

    def test_generate_data_creates_file(self):
        """Test if the data generation function creates an output file."""
        generate_data(output_dir=self.output_dir)
        self.assertTrue(os.path.exists(self.output_path))

    def test_generate_data_dataframe_not_empty(self):
        """Test if the generated dataframe is not empty."""
        df = generate_data(output_dir=self.output_dir)
        self.assertFalse(df.empty)

    def test_generate_data_has_expected_columns(self):
        """Test if the generated dataframe has the expected columns."""
        df = generate_data(output_dir=self.output_dir)
        expected_columns = [
            "duracion", "bpm", "energia", "acustica",
            "popularidad", "instrumentalidad", "genero_musical"
        ]
        self.assertListEqual(list(df.columns), expected_columns)

if __name__ == '__main__':
    unittest.main()
