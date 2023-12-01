import vapoursynth as vs
core = vs.core

core.max_cache_size = 40*1024
core.num_threads = 12

from adptvgrnMod import sizedgrn
from awsmfunc import bandmask
from havsfunc import SMDegrain
from muvsfunc import SSIM_downsample
from rekt import rektlvls
from vsutil import depth

###SOURCE
src = core.lsmas.LWLibavSource(r"B:\Yawara! BD BOX 1-3\[BDMV][140820][Yawara!][BD-BOX 1]\Disc 1\BDMV\STREAM\00000.m2ts").std.Crop(240, 240, 0, 0)
src16 = depth(src, 16).std.SetFrameProp(prop = "_Matrix", intval = 1)


###DENOISE
lumadown = SSIM_downsample(src16, w = 958, h = 720, sigmoid = True, filter_param_a = -1/2, filter_param_b = 1/4).resize.Bicubic(format=vs.YUV420P16, matrix = 1)
chromadown = core.fmtc.resample(src16, 958, 720, kernel = 'blackman')

lumamix = core.std.Merge(lumadown, chromadown, 0.35)
merge = core.std.ShufflePlanes([lumamix, chromadown], [0, 1, 2], vs.YUV)
up = core.resize.Spline36(merge, 1440, 1080)

cden1 = SMDegrain(merge, tr = 2, thSAD=1, thSADC = 250, RefineMotion=True, chroma = True)
cden1 = core.std.ShufflePlanes([merge, cden1], [0, 1, 2], vs.YUV)

cden2 = core.knlm.KNLMeansCL(merge, d = 4, a = 4, s = 3, h = 0.4, channels = 'uv', rclip = cden1)
cden2 = core.std.ShufflePlanes([merge, cden2], [0, 1, 2], vs.YUV)
#adjust cden strengths

###GRAIN
grainmask = bandmask(cden2, 600, pix=12, dec=2, exp=2).std.Convolution([1]*9)

grain1 = sizedgrn(cden2, strength = 0.3, cstrength = 0, size = 1.1, sharp = 90, static = False, grain_chroma = False, temporal_average = 0)
grain1 = core.std.MaskedMerge(cden2, grain1, grainmask)

grain2 = sizedgrn(cden2, strength = 0.5, cstrength = 0, size = 1.1, sharp = 90, static = False, grain_chroma = False, temporal_average = 0)
grain2 = core.std.MaskedMerge(cden2, grain2, grainmask)

grain3 = sizedgrn(cden2, strength = 0.7, cstrength = 0, size = 1.1, sharp = 90, static = False, grain_chroma = False, temporal_average = 0)
grain3 = core.std.MaskedMerge(cden2, grain3, grainmask)


#need perlin noise


###OUTPUT
out = depth(cden2, 10)

cden2.set_output()
grain1.set_output(3)
grain2.set_output(4)
grain3.set_output(5)