import unittest
from scripts.select_campaign import choose


class SelectionTests(unittest.TestCase):
    def test_stability_guard_and_recall_tie_band(self):
        rows=[{'id':'a','min_f1':.8,'min_recall':.6,'min_stability':.99,'max_background':.01,'mean_seconds':.1},
              {'id':'b','min_f1':.79,'min_recall':.8,'min_stability':.99,'max_background':.02,'mean_seconds':.2},
              {'id':'c','min_f1':.9,'min_recall':.9,'min_stability':.95,'max_background':.01,'mean_seconds':.1}]
        winner,passed=choose(rows,{'stability_guard':.98,'f1_band':.02})
        self.assertTrue(passed);self.assertEqual(winner['id'],'b')
        winner,passed=choose(rows,{'stability_guard':1.,'f1_band':.02})
        self.assertFalse(passed);self.assertEqual(winner['id'],'c')


if __name__=='__main__':unittest.main()
