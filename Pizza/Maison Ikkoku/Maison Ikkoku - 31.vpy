import vapoursynth as vs
core = vs.core

core.max_cache_size = 40*1024
core.num_threads = 12

from havsfunc import SMDegrain
from muvsfunc import SSIM_downsample
from vsutil import depth, get_w, plane


###SOURCE
src = core.lsmas.LWLibavSource(r"B:\めぞん一刻\MAISON_IKKOKU_06\BDMV\STREAM\00001.m2ts").resize.Point(src_left=1, resample_filter_uv="bicubic").std.Crop(240, 242, 0, 0)
src16 = depth(src, 16)


###DENOISE
lumadown = SSIM_downsample(src16, w = 958, h = 720, sigmoid = True, filter_param_a = -1/2, filter_param_b = 1/4).resize.Bicubic(format=vs.YUV420P16, matrix = 1)
chromadown = core.fmtc.resample(src16, 958, 720, kernel = 'blackman')

lumamix = core.std.Merge(lumadown, chromadown, 0.35)
merge = core.std.ShufflePlanes([lumamix, chromadown], [0, 1, 2], vs.YUV)

cden = SMDegrain(merge, tr = 4, thSAD=1, thSADC = 300, RefineMotion=True, chroma = True)
cden = core.std.ShufflePlanes([merge, cden], [0, 1, 2], vs.YUV)


###OUTPUT
out = depth(cden, 10)

out.set_output()
#plane(merge, 1).set_output(3)
#cden2.set_output(4)
#compare.set_output(5)