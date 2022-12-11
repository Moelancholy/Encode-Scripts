import vapoursynth as vs
core = vs.core

core.max_cache_size = 40*1024
core.num_threads = 12

from awsmfunc import bbmod
from havsfunc import SMDegrain
from muvsfunc import SSIM_downsample
from vsutil import depth, get_w, plane
from rekt import rektlvls


###SOURCE
compare = core.lsmas.LWLibavSource(r"E:\Anime\Kihachirou Kawamoto + Tadanari Okamoto Selected Films JP BDMV\Kihachirou Kawamoto BD\BDMV\STREAM\00003.m2ts").resize.Point(src_left=1, resample_filter_uv="bicubic").std.Crop(218,220,0,0)
src = core.lsmas.LWLibavSource(r"E:\Anime\Kihachirou Kawamoto + Tadanari Okamoto Selected Films JP BDMV\UHD BD\BDMV\STREAM\00003.m2ts").std.Crop(438,438,0,0)
src16 = depth(src, 16)


###FIX BORDERS
levels = rektlvls(src16, colnum = [0, 1, 2962, 2963], colval = [48, 3, 3, 46])
borders = bbmod(levels, 0, 0, 2, 2, thresh = 8, blur = [300, 700, 700], planes = [0, 1, 2])


###DENOISE
lumadown = SSIM_downsample(borders, w = 1482, h = 1080, sigmoid = True, filter_param_a = -1/2, filter_param_b = 1/4).resize.Bicubic(format=vs.YUV420P16, matrix = 1)
chromadown = core.fmtc.resample(borders, 1482, 1080, kernel = 'blackman')

lumamix = core.std.Merge(lumadown, chromadown, 0.35)
merge = core.std.ShufflePlanes([lumamix, chromadown], [0, 1, 2], vs.YUV)

cden = SMDegrain(merge, tr = 4, thSAD=1, thSADC = 250, RefineMotion=True, chroma = True)
cden = core.std.ShufflePlanes([merge, cden], [0, 1, 2], vs.YUV)


###OUTPUT
out = depth(cden, 10)

out.set_output()
merge.set_output(3)
#borders.set_output(4)
#cden1.set_output(5)
#cden2.set_output(6)
#cden3.set_output(7)