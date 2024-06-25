import unittest
from src.data_loader import load_data
from src.model import INSTRYModel

class TestINSTRYModel(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/industry_data.csv')
        self.model = INSTRYModel(self.data)

    def test_predict(self):
        result = self.model.predict("Python, machine learning, data analysis")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)

if __name__ == '__main__':
    unittest.main()
