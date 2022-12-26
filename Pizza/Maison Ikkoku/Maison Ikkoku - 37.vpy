import vapoursynth as vs
core = vs.core

from havsfunc import SMDegrain
from stgfunc import source
from vskernels import BicubicDidee
from vsscale import ssim_downsample
from vsutil import depth


###SOURCE
src = source(r".\Source\めぞん一刻\MAISON_IKKOKU_07\BDMV\STREAM\00000.m2ts").resize.Point(src_left=1, resample_filter_uv="bicubic").std.Crop(240, 242, 0, 0)
src16 = depth(src, 16)


###DENOISE
lumadown = ssim_downsample(src16, width = 958, height = 720, smooth = 1, scaler = BicubicDidee).resize.Bicubic(format=vs.YUV420P16, matrix = 1)
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