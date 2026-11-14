import unittest
from cvat_agent.ui import filter_labels
from cvat_agent.yolo_infer import label_mapping
class LabelSelectionTests(unittest.TestCase):
    def test_explicit_labels_do_not_expand_aliases(self):
        names=['traffic light','traffic_light','traffic sign','traffic_sign','road','lane/road curb','car']
        descriptions=[{'label':n} for n in names]
        self.assertEqual(filter_labels('traffic light, traffic sign', descriptions), [0,2])
        self.assertEqual(filter_labels('traffic_light, traffic_sign', descriptions), [1,3])
        self.assertEqual(filter_labels('lane/road curb', descriptions), [5])
    def test_yolo_prefers_exact_name(self):
        self.assertEqual(label_mapping({0:'traffic light'},['traffic light','traffic_light']),{0:'traffic light'})
        self.assertEqual(label_mapping({0:'traffic light'},['traffic_light']),{0:'traffic_light'})
