import unittest
import numpy as np
from cvat_agent.polyline import mask_polylines

class PolylineTests(unittest.TestCase):
    def test_disconnected_stripes_stay_separate(self):
        mask=np.zeros((20,30),bool)
        mask[3:6,2:12]=True
        mask[12:15,18:28]=True
        lines=mask_polylines(mask)
        self.assertEqual(len(lines),2)
        for points in lines:
            self.assertNotEqual(points[:2],points[-2:])
            for x,y in zip(points[::2],points[1::2]):
                self.assertTrue(mask[int(y),int(x)])

    def test_diagonal_and_limit(self):
        mask=np.eye(15,dtype=bool)
        lines=mask_polylines(mask,2)
        self.assertEqual(len(lines),1)
        self.assertEqual(len(lines[0]),4)
        self.assertEqual(mask_polylines(np.zeros((3,3),bool)),[])

    def test_branch_is_split(self):
        mask=np.zeros((11,11),bool)
        mask[1:10,5]=True
        mask[5,5:10]=True
        self.assertEqual(len(mask_polylines(mask)),3)
