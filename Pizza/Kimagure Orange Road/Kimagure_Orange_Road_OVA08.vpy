import vapoursynth as vs

import xvs

import havsfunc as haf
import lvsfunc as lvf
import vardefunc as vdf

from adptvgrnMod import adptvgrnMod
import G41Fun as gf
from finedehalo import fine_dehalo
from thresholding import kimagure_thresholding
from vsdpir import DPIR
from vsutil import depth, get_y

core = vs.core


###STANDARD IMPORT + CROP
src = core.lsmas.LWLibavSource(r"E:\Anime\KIMAGURE ORANGE ROAD OVA\BDMV\STREAM\00007.m2ts")
crop = depth(core.std.CropRel(src, 240, 240, 0, 0), 16)
crop_yuv = core.resize.Bicubic(crop, 720, 540, format=vs.YUV444P16)

edstart = 34344


###GENERATING MASKS (EPISODE)
preden = core.knlm.KNLMeansCL(crop_yuv, h=0.5, a=2, d=2, s=2, device_type='gpu', device_id=0, channels = 'YUV').resize.Bicubic(1440, 1080, format=vs.YUV420P16)
preden = core.std.ShufflePlanes([preden, crop], [0, 1, 2], vs.YUV)
mask0 = vdf.mask.PrewittStd().get_mask(get_y(preden), multi=2.00).std.Minimum().std.Maximum().std.Maximum()
mask1 = mask0.std.Binarize(40<<7).rgvs.RemoveGrain(20).rgvs.RemoveGrain(20)
mask2 = mask0.std.Binarize(70<<8)
mask3 = mask2.rgvs.RemoveGrain(20).rgvs.RemoveGrain(20)


###DENOISE CHROMA
crop444 = depth(vdf.scale.to_444(crop, 1440, 1080, join_planes = True), 32)
cropRGB = core.resize.Bicubic(crop444, format = vs.RGBS)

cden1 = core.ccd.CCD(cropRGB, 8)
cden2 = core.ccd.CCD(cropRGB, 12)

cden_thr = kimagure_thresholding(cropRGB, cden1, cden2)
cden_thr = core.std.ShufflePlanes([crop, core.resize.Bicubic(cden_thr, format = vs.YUV420P16, matrix = 1)], [0, 1, 2], vs.YUV)

luma_mask = core.adg.Mask(crop.std.PlaneStats(), luma_scaling=15)
cden_scaled = core.std.MaskedMerge(cden_thr, crop, luma_mask)


##FIX CHROMA
#crest1 = xvs.mwcfix(cden_scaled, kernel = 1, restore = 4, a = 2, grad = 4, warp = 3, thresh = 96, blur = 3, repair = 1)
#crest2 = xvs.mwcfix(cden_scaled, kernel = 1, restore = 4, a = 2, grad = 4, warp = 4, thresh = 128, blur = 3, repair = 1)

#crest1 = core.std.MaskedMerge(cden_scaled, crest1, mask1)
#crest = core.std.MaskedMerge(crest1, crest2, mask2)


##DERING
ring_mask = fine_dehalo(cden_scaled, rx = 2.1, thlimi = 50, thlima = 100, showmask = 1).rgvs.RemoveGrain(20).rgvs.RemoveGrain(20)
preden2 = core.knlm.KNLMeansCL(core.resize.Bicubic(cden_scaled, 1440, 1080, format=vs.YUV444P16), h=0.5, a=2, d=2, s=2, device_type='gpu', device_id=0, channels = 'YUV').resize.Bicubic(1440, 1080, format=vs.YUV420P16)
preden2 = core.std.ShufflePlanes([preden2, cden_scaled], [0, 1, 2], vs.YUV)
preden2 = core.std.MaskedMerge(cden_scaled, preden2, ring_mask)
diff = core.std.MakeDiff(cden_scaled, preden2)

dering = fine_dehalo(cden_scaled, rx = 2.1, thlimi = 50, thlima = 100, darkstr = 0, brightstr = 0.88, showmask = 0)
line_grain = adptvgrnMod(dering, strength=0.30, size=1, luma_scaling=1, static=False, grain_chroma = False)
dering_m = core.std.MaskedMerge(dering, line_grain, ring_mask)
dering_m = dering[:2652] + dering_m[2652:edstart] + dering[edstart:]
dering_merge = core.std.MergeDiff(dering_m, diff)


###OUPUTTING
grain = adptvgrnMod(dering_merge, strength=0.10, size=1.00, luma_scaling=15, sharp=60, static=False, grain_chroma = False)
out = depth(grain, 10)
cut = out[2653:2782] + out[4322:4517] + out[4751:4952] + out[6096:6130] + out[9720:10096] + out[10893:11000] + out[17991:18102] + out[21266:21505] + out[29780:30089]
out.set_output()
#dering_merge.set_output(2)
#out.set_output(3)
#cden_scaled1.set_output(3)
#cden_scaled2.set_output(4)
#cden_scaled3.set_output(5)
#dering.set_output(6)

#720 3491
#34175 36132