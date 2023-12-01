import vapoursynth as vs
core = vs.core

core.max_cache_size = 40*1024
core.num_threads = 12

from awsmfunc import bbmod
from havsfunc import SMDegrain
from muvsfunc import SSIM_downsample
from rekt import rektlvls
from vsutil import depth

###SOURCE
src = core.lsmas.LWLibavSource(r"B:\Yawara! BD BOX 1-3\[BDMV][140820][Yawara!][BD-BOX 1]\Disc 1\BDMV\STREAM\00000.m2ts").std.Crop(240, 240, 0, 0)
src16 = depth(src, 16).std.SetFrameProp(prop = "_Matrix", intval = 1)


###DENOISE
#lumadown = SSIM_downsample(src16, w = 958, h = 720, sigmoid = True, filter_param_a = -1/2, filter_param_b = 1/4).resize.Bicubic(format=vs.YUV420P16, matrix = 1)
#chromadown = core.fmtc.resample(src16, 958, 720, kernel = 'blackman')

#lumamix = core.std.Merge(lumadown, chromadown, 0.35)
#merge = core.std.ShufflePlanes([lumamix, chromadown], [0, 1, 2], vs.YUV)
#up = core.resize.Spline36(merge, 1440, 1080)

cden1 = SMDegrain(src16, tr = 2, thSAD=1, thSADC = 250, RefineMotion=True, chroma = True)
cden1 = core.std.ShufflePlanes([src16, cden1], [0, 1, 2], vs.YUV)

cden2 = core.knlm.KNLMeansCL(src16, d = 4, a = 4, s = 3, h = 0.4, channels = 'uv', rclip = cden1)
cden2 = core.std.ShufflePlanes([src16, cden2], [0, 1, 2], vs.YUV)


###OUTPUT
out = depth(cden2, 10)

out.set_output()
#cden1.set_output(3)
#cden2.set_output(4)
#cden3.set_output(5)