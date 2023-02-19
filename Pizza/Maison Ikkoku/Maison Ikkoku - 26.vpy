from vstools import vs, core, depth
from muvsfunc import SSIM_downsample
from vsdenoise import  mlm_degrain, Prefilter, SADMode, SearchMode
from vskernels import BlackManMinLobe, Spline36

core.set_affinity(range(14, 24), 30000)


###SOURCE
src = core.lsmas.LWLibavSource(r"A:\めぞん一刻\MAISON_IKKOKU_05\BDMV\STREAM\00001.m2ts").resize.Point(src_left=1, resample_filter_uv="bicubic").std.Crop(240, 242, 0, 0)
src16 = depth(src, 16)


###DENOISE
lumadown = SSIM_downsample(src16, w = 958, h = 720, sigmoid = True, filter_param_a = -1/2, filter_param_b = 1/4).resize.Bicubic(format=vs.YUV420P16, matrix = 1)
chromadown = core.fmtc.resample(src16, 958, 720, kernel = 'blackman')

lumamix = core.std.Merge(lumadown, chromadown, 0.35)
merge = core.std.ShufflePlanes([lumamix, chromadown], [0, 1, 2], vs.YUV)

cdenu1 = mlm_degrain(merge, tr = 3, refine = 4, thSAD=40, scaler = Spline36, downscaler=BlackManMinLobe, planes=1, prefilter = Prefilter.DFTTEST, search = SearchMode.DIAMOND, sad_mode = SADMode.ADAPTIVE_SATD_MIXED)
cdenu2 = mlm_degrain(cdenu1, tr = 3, refine = 4, thSAD=120, scaler = Spline36, downscaler=BlackManMinLobe, planes=1, prefilter = Prefilter.BILATERAL, search = SearchMode.DIAMOND, sad_mode = SADMode.ADAPTIVE_SATD_MIXED)
cdenv2 = core.knlm.KNLMeansCL(merge, d = 4, a = 4, s = 3, h = 0.25, channels = 'uv')

cden = core.std.ShufflePlanes([merge, cdenu2, cdenv2], [0,1,2], vs.YUV)


###OUTPUT
out = depth(cden, 10)

out.set_output()
#plane(merge, 1).set_output(3)
#cden2.set_output(4)
#compare.set_output(5)