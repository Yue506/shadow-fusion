import unittest
import numpy as np
from shadow_fusion import compose, make_config
from shadow_fusion.research import compose_research


class ResearchTests(unittest.TestCase):
    def test_association_rejects_disconnected_change_and_preserves_defaults(self):
        source=np.full((128,128,3),90,dtype='uint8')
        generated=source.copy();generated[80:110,60:78]=130;generated[12:32,12:32]=130
        mask=np.zeros((128,128),bool);mask[50:80,50:80]=True
        base=compose(source,generated,mask,make_config(color=False))
        equivalent=compose_research(source,generated,mask,{'color':False})
        for key in ['output','alpha','support','envelope','corrected']:
            np.testing.assert_array_equal(base[key],equivalent[key])
        connected=compose_research(source,generated,mask,{'color':False,'association':'connected'})
        self.assertTrue(connected['support'][90,65])
        self.assertTrue(base['support'][20,20])
        self.assertFalse(connected['support'][20,20])
        np.testing.assert_array_equal(source[~connected['envelope']],connected['output'][~connected['envelope']])
        no_extra=compose_research(source,generated,mask,{'extra':False,'association':'connected'})
        expected=compose(source,generated,mask,make_config(extra=False))
        np.testing.assert_array_equal(no_extra['output'],expected['output'])


if __name__=='__main__':unittest.main()
