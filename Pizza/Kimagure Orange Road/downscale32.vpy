import vapoursynth as vs

import xvs

import havsfunc as haf
import lvsfunc as lvf
import vardefunc as vdf

from adptvgrnMod import adptvgrnMod
from muvsfunc import SSIM_downsample
from thresholding import kimagure_thresholding
from vsdpir import DPIR
from vsutil import depth, get_y

core = vs.core


###STANDARD IMPORT + CROP
src = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV4\BDMV\STREAM\00001.m2ts")
op = core.lsmas.LWLibavSource(r"C:\Users\twint\Documents\GitHub\Encode-Scripts\Pizza\Kimagure Orange Road\KOR_OP2_720_lossless.mkv")
crop = depth(core.std.CropRel(src, 240, 240, 0, 0), 16)
crop_yuv = core.resize.Bicubic(crop, 720, 540, format=vs.YUV444P16)

edstart = 33768
preview = 36144


###GENERATING MASKS (EPISODE)
preden = core.knlm.KNLMeansCL(crop_yuv, h=1.0, a=2, d=2, s=2, device_type='gpu', device_id=0, channels = 'YUV').resize.Bicubic(1440, 1080, format=vs.YUV420P16)
preden = core.std.ShufflePlanes([preden, crop], [0, 1, 2], vs.YUV)
mask0 = vdf.mask.PrewittStd().get_mask(get_y(preden), multi=2.00).std.Minimum().std.Maximum().std.Maximum().resize.Bicubic(960, 720)
mask1 = mask0.std.Binarize(40<<7).rgvs.RemoveGrain(20).rgvs.RemoveGrain(20).resize.Bicubic(960, 720)
mask2 = mask0.std.Binarize(70<<8).resize.Bicubic(960, 720)


###FIX CHROMA
crop444 = depth(vdf.scale.to_444(crop, 1440, 1080, join_planes = True), 32)
cropRGB = core.resize.Bicubic(crop444, format = vs.RGBS)
cropRGB_down_y = SSIM_downsample(cropRGB, 960, 720, smooth = ((3 ** 2 - 1) / 12) ** 0.5, sigmoid = True, filter_param_a=-1/2, filter_param_b=1/4)
cropRGB_down_y = get_y(core.resize.Bicubic(cropRGB_down_y, format = vs.YUV420P16, matrix = 1))
cropRGB_down_uv = core.resize.Bicubic(cropRGB, 960, 720)
alt_uv = core.resize.Bicubic(cropRGB_down_uv, matrix = 1, format = vs.YUV420P16)
alt = core.std.ShufflePlanes([cropRGB_down_y, alt_uv], [0, 1, 2], vs.YUV)

cden1 = core.ccd.CCD(cropRGB_down_uv, 20)
cden2 = core.ccd.CCD(cropRGB_down_uv, 30)

cden_thr = kimagure_thresholding(cropRGB_down_uv, cden1, cden2)
cden_thr = core.std.ShufflePlanes([cropRGB_down_y, core.resize.Bicubic(cden_thr, format = vs.YUV420P16, matrix = 1)], [0, 1, 2], vs.YUV)
cden_test = core.std.ShufflePlanes([cropRGB_down_y, core.resize.Bicubic(cden2, format = vs.YUV420P16, matrix = 1)], [0, 1, 2], vs.YUV)

crest1 = xvs.mwcfix(cden_thr, kernel = 1, restore = 5, a = 2, grad = 4, warp = 4, thresh = 128, blur = 2, repair = 1)
crest2 = xvs.mwcfix(cden_thr, kernel = 2, restore = 5, a = 2, grad = 4, warp = 4, thresh = 140, blur = 2, repair = 1)
crest = core.std.MaskedMerge(crest1, crest2, mask2)


##ED SCENEFILTERING
edcrop = alt[edstart:preview]
ed_crest1 = xvs.mwcfix(edcrop, kernel = 1, restore = 5, a = 2, grad = 4, warp = 6, thresh = 128, blur = 3, repair = 1)
ed_crest2 = xvs.mwcfix(edcrop, kernel = 2, restore = 5, a = 2, grad = 4, warp = 6, thresh = 140, blur = 3, repair = 1)
ed_crest = core.std.MaskedMerge(ed_crest1, ed_crest2, mask2)

ed_444 = depth(vdf.scale.to_444(ed_crest, width = 960, height = 720, join_planes = True), 32)
ed_rgb = core.resize.Bicubic(ed_444, format = vs.RGBS)
deblock = DPIR(ed_rgb, strength = 12, task = 'deblock', device_type = 'cpu')
deblock = core.resize.Bicubic(deblock, format = vs.YUV420P16, matrix = 1, dither_type = 'error_diffusion')
debl_grain = adptvgrnMod(deblock, strength=0.10, size=1.40, luma_scaling=1, sharp=40, static=False, grain_chroma = False, seed = 69)
luma_mask = core.adg.Mask(ed_crest.std.PlaneStats(), luma_scaling = 18)
deblock_dark = core.std.MaskedMerge(ed_crest, debl_grain, luma_mask)
deblock_dark = adptvgrnMod(deblock_dark, strength=0.08, size=1.40, luma_scaling=1, sharp=40, static=False, grain_chroma = False, seed = 69)
ed_crest = adptvgrnMod(ed_crest, strength=0.08, size=1.40, luma_scaling=5, sharp=40, static=False, grain_chroma = False, seed = 69)
ed_deblock = depth(lvf.rfs(ed_crest, deblock_dark, [(132, 278), (519, 622)]), 10)



##DERING
dering = haf.EdgeCleaner(crest, strength = 4, rep = True, rmode = 17, smode = 1, hot = True)
dering_m = core.std.MaskedMerge(crest, dering, mask1)


###OUPUTTING
episode = depth(dering_m, 10)
out = episode[:720] + op + episode[3012:edstart] + ed_deblock + episode[preview:]
#cut = out[5876:5942] + out[13182:13269] + out[13792:13960] + out[14197:14492] + out[14623:14864] + out[14923:15109] + out[20298:20487] + out[20800:20958] + out[12726:12948] + out[31369:31549]
#cut.set_output()
#before = core.std.ShufflePlanes([cropRGB_down_y, core.resize.Bicubic(cropRGB_down_uv, format = vs.YUV420P16, matrix = 1)], [0, 1, 2], vs.YUV)
#ed_crest.set_output(1)
#deblock_dark.set_output(2)
#crest.set_output(3)
out.set_output()

#2999
#34188 36128
