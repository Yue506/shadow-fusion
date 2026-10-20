import unittest
import numpy as np
from scipy import ndimage as ndi
from shadow_fusion import compose, make_config
from shadow_fusion.research import compose_research,calibrate_srgb


class ResearchTests(unittest.TestCase):
    def test_bounded_feather_preserves_outside_support_and_object_edit(self):
        source=np.full((128,128,3),80,dtype='uint8');g=source.copy()
        g[45:100,50:85]=140
        mask=np.zeros((128,128),bool);mask[45:70,50:85]=True
        for feather in ['adaptive','inward','inward_multilevel']:
            result=compose_research(source,g,mask,{'color':False,'feather':feather})
            np.testing.assert_array_equal(result['output'][mask],g[mask])
            np.testing.assert_array_equal(result['output'][~result['envelope']],source[~result['envelope']])
            if feather.startswith('inward'):
                np.testing.assert_array_equal(result['envelope'],result['support'])
                boundary=result['support'] & ~ndi.binary_erosion(result['support']) & ~mask
                self.assertLess(result['alpha'][boundary].max(),.5)

    def test_plane_fit_recovers_background_without_erasing_removed_shadow(self):
        rng=np.random.default_rng(42)
        source=rng.integers(40,180,(128,128,3),dtype='uint8')
        yy,xx=np.indices((128,128));plane=4*(xx/127-.5)-3*(yy/127-.5)
        candidate=source.astype(float).copy();candidate[70:100,60:85]+=35
        candidate=np.rint(candidate*np.array([1.02,.98,1.01])+[5,-3,2]+plane[...,None]).clip(0,255).astype('uint8')
        mask=np.zeros((128,128),bool);mask[40:65,50:80]=True
        shadow=np.zeros_like(mask);shadow[70:100,60:85]=True
        corrected,info=calibrate_srgb(source,candidate,mask,spatial=True)
        self.assertEqual(info['status'],'fit')
        self.assertLess(np.abs(corrected.astype(float)-source)[~(mask|shadow)].mean(),.7)
        self.assertGreater((corrected.astype(float)-source)[shadow].mean(),30)

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
