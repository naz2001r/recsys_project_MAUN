import unittest
import sys
sys.path.append('./src/metrics/')
from metrics import ReccomenderMetrics

class MetricsTests(unittest.TestCase):

    def test_apk_should_return_correct_value(self):
        relevant_items = ['Nazarii', 'Uliana']
        all_items = ['Mykola', 'Ivan', 'Oleksandr', 'Nazarii', 'Uliana']
        metrics = ReccomenderMetrics()
        assert round(metrics.apk(relevant_items, all_items, k=5), 2) == 0.33

    def test_mapk_should_return_correct_value(self):
        relevant_items = ['Nazarii', 'Uliana']
        all_items = ['Mykola', 'Ivan', 'Oleksandr', 'Marianna', 'Nazarii', 'Uliana']
        metrics = ReccomenderMetrics()
        assert round(metrics.apk(relevant_items, all_items, k=6), 3) == 0.267

    def test_mapk_should_return_correct_value2(self):
        relevant_items = ['Nazarii', 'Uliana']
        all_items = ['Nazarii', 'Uliana', 'Mykola', 'Ivan', 'Oleksandr', 'Marianna']
        metrics = ReccomenderMetrics()
        assert round(metrics.apk(relevant_items, all_items, k=6), 3) == 1

    def test_mapk_should_return_correct_value_multi_users(self):
        relevant_items = [['Nazarii', 'Uliana'], ['Nazarii', 'Uliana'], ['Nazarii', 'Uliana']]
        all_items = [['Nazarii', 'Uliana', 'Mykola', 'Ivan', 'Oleksandr', 'Marianna'], \
                    ['Mykola', 'Ivan', 'Oleksandr', 'Marianna', 'Nazarii', 'Uliana'], \
                    ['Mykola', 'Nazarii', 'Ivan', 'Uliana', 'Oleksandr', 'Marianna']]
        metrics = ReccomenderMetrics()
        assert round(metrics.mapk(relevant_items, all_items, k=6), 2) == 0.59


if __name__=='__main__':
	unittest.main()