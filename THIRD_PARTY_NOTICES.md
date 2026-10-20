# Sources and attribution

- TraceFusion concepts and parameters: *LensErase: A Foundation Model for Erase Anything*, §3.3 and Appendix E.5, https://arxiv.org/abs/2610.18072 . The implementation here was written independently from the paper. It is not official LensErase code and does not include LensErase weights.
- Existing local research used DESOBA visualizations: https://github.com/bcmi/Object-Shadow-Generation-Dataset-DESOBA . Dataset references and underlying image rights remain with their respective authors. Original dataset images and masks are not redistributed in this repository.
- Existing controlled local fixtures also used textures from https://github.com/mrdoob/three.js . Those texture images and the original photo cutouts are not redistributed here.
- NumPy, SciPy and Pillow are installed dependencies, not vendored source. Their own licenses apply to their distributions.

The small committed `examples/` images and `docs/assets/example.png` are deterministic, code-generated gradient fixtures. Their candidate is a known clean image with an injected color bias, not a removal model result.

The project MIT license applies to original project code and the synthetic examples, not to third-party datasets, model weights or source papers.
# Optional CPU object proposals

The exploration-only ownership module can download OpenCV Zoo YOLOX_s and its
official Python wrapper, licensed under Apache-2.0. These files are not bundled
or relicensed as project source. The download keeps the upstream LICENSE and
verifies fixed SHA-256 checksums. Upstream revision:
`opencv/opencv_zoo@48b10fa505aeb51fbd127fc60a5c660040e823b7`, directory
`models/object_detection_yolox/`.

https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox

