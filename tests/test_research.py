import unittest
import json
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage as ndi
from shadow_fusion import compose, make_config
from shadow_fusion.research import compose_research,calibrate_srgb,local_anchor_correction,photometric_guard,luminance_transfer,luma_offset_transfer,ResearchConfig
from shadow_fusion.cli import main


class ResearchTests(unittest.TestCase):
    def test_guard_rejects_recolour_but_accepts_scalar_illumination(self):
        source=np.full((96,96,3),80,dtype='uint8');g=source.copy()
        g[35:80,35:60]=130;g[35:80,60:80]=[190,30,60]
        gate=photometric_guard(source,g,'photometric')
        self.assertGreater(gate[55,45],.99);self.assertEqual(gate[55,70],0)
        mask=np.zeros((96,96),bool);mask[20:40,35:60]=True
        result=compose_research(source,g,mask,{'color':False,'guard':'photometric','confidence':'positive','feather':'compact'})
        np.testing.assert_array_equal(result['output'][50:70,65:75],source[50:70,65:75])
        np.testing.assert_array_equal(result['output'][mask],g[mask])
        self.assertGreater(result['alpha'][55,45],.99)
        self.assertEqual(result['metadata']['outside_envelope_max_channel_difference'],0)

    def test_luminance_transfer_preserves_source_detail_and_mask_edit(self):
        source=np.full((64,64,3),80,dtype='uint8');source[::2,::2]=85
        candidate=np.full_like(source,130);mask=np.zeros((64,64),bool);mask[20:30,20:30]=True
        output=luminance_transfer(source,candidate,mask)
        np.testing.assert_array_equal(output[mask],candidate[mask])
        self.assertGreater(int(output[10,10,0]),int(output[11,10,0]))
        np.testing.assert_array_equal(output[...,0],output[...,1])
        offset=luma_offset_transfer(source,candidate,mask)
        np.testing.assert_array_equal(offset[mask],candidate[mask])
        self.assertGreater(int(offset[10,10,0]),int(offset[11,10,0]))
        for kwargs in [{'guard':'invalid'},{'transfer':'invalid'}]:
            with self.assertRaises(ValueError):ResearchConfig(**kwargs)

    def test_relative_gate_suppresses_small_bright_background_drift(self):
        s=np.full((96,96,3),180,dtype='uint8');g=s.copy();g[45:75,30:55]+=6
        m=np.zeros((96,96),bool);m[25:45,30:55]=True
        s[75:90,30:55]=50;g[75:90,30:55]=100
        result=compose_research(s,g,m,{'color':False,'confidence':'relative','feather':'compact'})
        np.testing.assert_array_equal(result['output'][55,40],s[55,40])
        np.testing.assert_array_equal(result['output'][80,40],g[80,40])

    def test_local_anchor_fit_reduces_nonplanar_bias_without_learning_shadow(self):
        yy,xx=np.indices((128,128));source=np.zeros((128,128,3),dtype='uint8')
        source[:]=[100,120,140]
        bias=3*np.cos(xx/127*2*np.pi)*np.cos(yy/127*2*np.pi)
        g=np.rint(source.astype(float)+bias[...,None]).astype('uint8')
        mask=np.zeros((128,128),bool);mask[50:70,50:70]=True
        shadow=np.zeros_like(mask);shadow[70:95,55:75]=True;g[shadow]+=30
        for material in [False,True]:
            corrected,info=local_anchor_correction(source,g,mask,material,strength=.5)
            bg=~(mask|shadow)
            self.assertEqual(info['status'],'fit')
            self.assertLess(np.abs(corrected.astype(float)-source)[bg].mean(),np.abs(g.astype(float)-source)[bg].mean())
            self.assertGreater((corrected.astype(float)-source)[shadow].mean(),25)
        unchanged,_=local_anchor_correction(source,g,mask,True,strength=0)
        np.testing.assert_array_equal(unchanged,g)

    def test_weak_path_connects_separated_shadow_without_adding_unrelated_one(self):
        source=np.full((128,128,3),80,dtype='uint8');g=source.copy()
        mask=np.zeros((128,128),bool);mask[30:50,50:70]=True
        g[50:62,55:67]+=2;g[62:85,55:67]+=40;g[80:105,95:115]+=40
        hard=compose_research(source,g,mask,{'color':False,'association':'connected'})
        weak=compose_research(source,g,mask,{'color':False,'association':'weak_connected','weak_low':1})
        self.assertFalse(hard['support'][75,60]);self.assertTrue(weak['support'][75,60])
        self.assertFalse(weak['support'][90,105])
        np.testing.assert_array_equal(weak['output'][~weak['envelope']],source[~weak['envelope']])

    def test_cli_research_flags_and_ablations_are_effective(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);source=np.full((64,64,3),90,dtype='uint8');g=source.copy()
            g[20:35,20:35]=140;mask=np.zeros((64,64),'uint8');mask[20:35,20:35]=255
            for name,a in [('s.png',source),('g.png',g),('m.png',mask)]:Image.fromarray(a).save(p/name)
            (p/'research.json').write_text(json.dumps({'calibration':'srgb_plane','confidence':'positive','association':'connected','guard':'photometric','transfer':'luminance'}))
            args=['compose','--source',str(p/'s.png'),'--generated',str(p/'g.png'),'--mask',str(p/'m.png'),
                  '--output',str(p/'out.png'),'--research-config',str(p/'research.json'),'--mode','shadow',
                  '--gamma','on','--ablate','color,extra,confidence,feather,guard,transfer']
            self.assertEqual(main(args),0)
            meta=json.loads((p/'out.json').read_text());config=meta['config']
            self.assertEqual(config['engine'],'research');self.assertFalse(config['color'])
            self.assertFalse(config['extra']);self.assertFalse(config['gamma'])
            self.assertEqual(config['confidence'],'none');self.assertEqual(config['feather'],'none')
            self.assertEqual(config['guard'],'none');self.assertEqual(config['transfer'],'rgb')
            result=np.array(Image.open(p/'out.png'))
            np.testing.assert_array_equal(result[mask>0],g[mask>0])

    def test_positive_gate_blocks_background_copy_but_keeps_shadow(self):
        source=np.full((128,128,3),90,dtype='uint8');g=source.copy()
        g[...,0]+=1;g[70:105,55:85]+=35;g[45:70,50:85]=130
        mask=np.zeros((128,128),bool);mask[45:70,50:85]=True
        target=np.zeros_like(mask);target[70:105,55:85]=True
        for feather in ['single','adaptive','multilevel','inward_multilevel']:
            result=compose_research(source,g,mask,{'color':False,'confidence':'positive','feather':feather})
            np.testing.assert_array_equal(result['output'][~(mask|target)],source[~(mask|target)])
            np.testing.assert_array_equal(result['output'][mask],g[mask])
            self.assertTrue((result['support']<=result['envelope']).all())
        ordinary=compose_research(source,g,mask,{'color':False,'confidence':'positive'})
        np.testing.assert_array_equal(ordinary['output'][85,65],g[85,65])

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
