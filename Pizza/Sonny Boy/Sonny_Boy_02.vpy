import vapoursynth as vs
import debandshit
from muvsfunc import SSIM_downsample

import G41Fun as gf
import havsfunc as haf
import lvsfunc as lvf
import kagefunc as kgf
import vardefunc as vdf
import zzfunc as zzf

from vsutil import *
core = vs.core



###SCENEFILTER
edstart = 33086

hardsubs_GTFO_ranges = [
    (2919, 2972), (3364, 3482), (4320, 4421),
    (5127, 5279), (5454, 5525), (5718, 5825),
    (7488, 7535), (11918, 12001), (12440, 12487),
    (30908, 31075)
]



MOVES_LIKE_JAGGED_RANGES = [
    (154, 214), (1802, 1942), (2210, 2257),
    (2330, 2371), (2537, 2764), (3483, 3605),
    (3939, 3974), (26650, 26743), (27542, 27725),
    (32781, 32918)
]



clamp_ranges = [
    (431, 562), (3364, 3482), (4083, 4118),
    (4320, 4421), (4791, 4898), (4983, 5126),
    (5454, 5525), (5718, 5825), (7449, 7487),
    (9376, 9426), (12440, 12487), (26206, 26493),
    (30012, 30082), (30908, 31075), (31217, 31279)
]



deband_ranges = [
    (8103, 8168), (21640, 21723), (24075, 24236),
    (25468, 25575), (25620, 25817)
]



strong_deband_ranges = [
    (19221, 19388), (19692, 19934), (20717, 21085),
    (21410, 21639), (23837, 23890), (24237, 24320),
    (24476, 24523), (24569, 24688), (24854, 24961),
    (25220, 25467), (25576, 25619), (25818, 25871),
    (28070, 28111)
]



###SOURCES
waka = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\Sonny Boy\Waka\Sonny Boy E02 [1080p][AAC][JapDub][GerSub][Web-DL].mkv")
waka = depth(waka, 16)
cbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\Sonny Boy\Amazon\Sonny Boy - 02 (Amazon Prime CBR 1080p).mkv")[12:-12]
cbr = depth(cbr, 16).std.Expr("x 128 +")
waka = waka[0] + waka

shader = r"C:\Users\twint\AppData\Roaming\mpv\shaders\FSRCNNX_x2_56-16-4-1.glsl"



###DEHARDSUB
sub_mask_1 = kgf.hardsubmask(waka, cbr)
sub_mask_2 =  core.std.Expr(split(core.std.Expr([waka, cbr], 'x y - abs').resize.Point(format=vs.YUV444P16)), 'x y z max max')
sub_mask_2 = sub_mask_2.std.Binarize(10<<8).std.Minimum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Inflate().std.Inflate().std.Inflate().std.Inflate().std.Inflate()
sub_mask_2 = zzf.combine(zzf.shiftframes(sub_mask_2, [-6, 6])).std.Expr('x 257 *', vs.GRAY16)
sub_mask = lvf.rfs(sub_mask_1, sub_mask_2, hardsubs_GTFO_ranges)
dehardsub = core.std.MaskedMerge(waka, cbr, sub_mask)
preden = haf.SMDegrain(get_y(dehardsub), tr=2, RefineMotion=True, thSAD=150)



###AA
upscale =  vdf.scale.fsrcnnx_upscale(get_y(dehardsub), width = 3840, height = 2160, downscaler = None, shader_file=shader, profile = 'zastin')
aa_mask = vdf.mask.FreyChen().get_mask(preden, lthr=0, multi=1.70).std.Binarize(62<<7).std.Maximum().rgvs.RemoveGrain(20)

weak_aa = depth(SSIM_downsample(upscale, w = 3200, h = 1800, sigmoid = True), 16)
weak_aa = lvf.aa._eedi3_singlerate(weak_aa).std.Transpose()
weak_aa = lvf.aa._eedi3_singlerate(weak_aa).std.Transpose()
weak_aa = depth(SSIM_downsample(weak_aa, w = 1920, h = 1080, sigmoid = True), 16)
aa1 = core.std.MaskedMerge(get_y(dehardsub), weak_aa, aa_mask)

strong_aa = depth(SSIM_downsample(upscale, w = 2880, h = 1620, sigmoid = True), 16)
strong_aa = lvf.aa._eedi3_singlerate(strong_aa).std.Transpose()
strong_aa = lvf.aa._eedi3_singlerate(strong_aa).std.Transpose()
strong_aa = depth(SSIM_downsample(strong_aa, w = 1920, h = 1080, sigmoid = True), 16)
aa2 = core.std.MaskedMerge(get_y(dehardsub), strong_aa, aa_mask)

nnedi3 = lvf.aa.taa(get_y(dehardsub), lvf.aa.nnedi3(opencl=True))
aa3 = lvf.aa.clamp_aa(get_y(dehardsub), nnedi3, aa2, strength = 3.0)
aa3 = core.std.MaskedMerge(get_y(dehardsub), aa3, aa_mask)

aa_y = lvf.rfs(aa1, aa2, MOVES_LIKE_JAGGED_RANGES)
aa_y = lvf.rfs(aa_y, aa3, clamp_ranges)
aa = core.std.ShufflePlanes([aa_y, dehardsub], [0, 1, 2], vs.YUV)



###DERING
dehalo = gf.MaskedDHA(aa, rx = 1.6, ry = 1.6, darkstr = 0.0, brightstr = 1.0, showmask = False)
dering = haf.EdgeCleaner(dehalo, strength = 6, rep = True, rmode = 13, smode = 1, hot = True)



###DEBAND
preden2 = haf.SMDegrain(dering, tr=2, RefineMotion=True, thSAD=150)
deband_mask = kgf.retinex_edgemask(preden2)
diff = core.std.MakeDiff(dering, preden2)

deband = debandshit.dumb3kdb(preden2, radius=16, threshold=40, grain=0)
deband = core.std.MergeDiff(deband, diff)
deband = core.std.MaskedMerge(deband, dering, deband_mask)

strong_deband = debandshit.dumb3kdb(preden2, radius=16, threshold=64, grain=0)
strong_deband = core.std.MergeDiff(strong_deband, diff)
strong_deband = core.std.MaskedMerge(strong_deband, dering, deband_mask)

flt = lvf.rfs(dering, deband, deband_ranges)
flt = lvf.rfs(flt, strong_deband, strong_deband_ranges)
flt = flt[:edstart] + dehardsub[edstart:]


###OUTPUT
out = depth(flt, 10)
out.set_output()
#waka.set_output(1)
#get_y(dehardsub).set_output(2)
#aa1.set_output(3)
#aa2.set_output(4)
#aa3.set_output(5)
#dering.set_output(6)
#deband.set_output(7)
#strong_deband.set_output(8)

