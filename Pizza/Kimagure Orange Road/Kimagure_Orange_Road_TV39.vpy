import vapoursynth as vs

import xvs

import havsfunc as haf
import lvsfunc as lvf
import vardefunc as vdf

from adptvgrnMod import adptvgrnMod
from thresholding import kimagure_thresholding
from vsdpir import DPIR
from vsutil import depth, get_y

core = vs.core


###STANDARD IMPORT + CROP
src = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV4\BDMV\STREAM\00008.m2ts")
op = core.lsmas.LWLibavSource(r"C:\Users\twint\Documents\GitHub\Encode-Scripts\Pizza\Kimagure Orange Road\KOR_OP3_lossless.mkv")[:-25]
crop = depth(core.std.CropRel(src, 240, 240, 0, 0), 16)
crop_yuv = core.resize.Bicubic(crop, 720, 540, format=vs.YUV444P16)


###GENERATING MASKS (EPISODE)
preden = core.knlm.KNLMeansCL(crop_yuv, h=1.0, a=2, d=2, s=2, device_type='gpu', device_id=0, channels = 'YUV').resize.Bicubic(1440, 1080, format=vs.YUV420P16)
preden = core.std.ShufflePlanes([preden, crop], [0, 1, 2], vs.YUV)
mask0 = vdf.mask.PrewittStd().get_mask(get_y(preden), multi=2.00).std.Minimum().std.Maximum().std.Maximum()
mask1 = mask0.std.Binarize(40<<7).rgvs.RemoveGrain(20).rgvs.RemoveGrain(20)
mask2 = mask0.std.Binarize(70<<8)


###FIX CHROMA
crop444 = depth(vdf.scale.to_444(crop, 1440, 1080, join_planes = True), 32)
cropRGB = core.resize.Bicubic(crop444, format = vs.RGBS)

cden1 = core.ccd.CCD(cropRGB, 20)
cden2 = core.ccd.CCD(cropRGB, 30)

cden_thr = kimagure_thresholding(cropRGB, cden1, cden2)
cden_thr = core.std.ShufflePlanes([crop, core.resize.Bicubic(cden_thr, format = vs.YUV420P16, matrix = 1)], [0, 1, 2], vs.YUV)
cden_test = core.std.ShufflePlanes([crop, core.resize.Bicubic(cden2, format = vs.YUV420P16, matrix = 1)], [0, 1, 2], vs.YUV)

crest1 = xvs.mwcfix(cden_thr, kernel = 1, restore = 5, a = 2, grad = 4, warp = 6, thresh = 128, blur = 3, repair = 1)
crest2 = xvs.mwcfix(cden_thr, kernel = 2, restore = 5, a = 2, grad = 4, warp = 6, thresh = 140, blur = 3, repair = 1)
crest = core.std.MaskedMerge(crest1, crest2, mask2)


##DERING
dering = haf.EdgeCleaner(crest, strength = 10, rep = True, rmode = 17, smode = 1, hot = True)
dering_m = core.std.MaskedMerge(crest, dering, mask1)


###OUPUTTING
episode = depth(dering_m, 10)
out = episode[:720] + op + episode[3324:]
#cut = out[5876:5942] + out[13182:13269] + out[13792:13960] + out[14197:14492] + out[14623:14864] + out[14923:15109] + out[20298:20487] + out[20800:20958] + out[12726:12948] + out[31369:31549]
#src.set_output()
out.set_output()

#720 3491
#34176 36143