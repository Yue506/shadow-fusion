import unittest
import numpy as np
from shadow_fusion.ownership import other_seeds,competitive_support


class OwnershipTests(unittest.TestCase):
    def test_target_box_is_exempt_and_inactive_competitor_has_no_seed(self):
        mask=np.zeros((64,96),bool);mask[10:30,10:25]=True
        boxes=np.array([[9,9,17,22,.9,0],[65,10,20,25,.8,0]],float)
        seeds,info=other_seeds(mask,boxes)
        self.assertEqual(info['matched_target_index'],0)
        self.assertEqual(info['other_boxes'],1)
        self.assertFalse((seeds&mask).any());self.assertTrue(seeds[33,75])
        inactive,info=other_seeds(mask,boxes,np.zeros(mask.shape))
        self.assertFalse(inactive.any());self.assertEqual(info['active_other_boxes'],0)

    def test_competition_keeps_target_shadow_and_rejects_other_shadow(self):
        mask=np.zeros((64,96),bool);mask[10:25,10:25]=True
        seeds=np.zeros_like(mask);seeds[22:25,70:80]=True
        support=mask.copy();support[24:50,15:30]=True;support[24:50,65:80]=True
        evidence=np.zeros(mask.shape);evidence[support&~mask]=40
        for method in ['euclidean','geodesic']:
            owned=competitive_support(mask,support,evidence,seeds,method,.9)
            self.assertTrue(owned[40,22]);self.assertFalse(owned[40,72])
            self.assertTrue((owned[mask]).all())
        self.assertIsNone(competitive_support(mask,support,evidence,np.zeros_like(mask)))


if __name__=='__main__':unittest.main()
