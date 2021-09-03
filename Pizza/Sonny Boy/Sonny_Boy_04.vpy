import vapoursynth as vs
import debandshit

import awsmfunc as awf
import G41Fun as gf
import havsfunc as haf
import lvsfunc as lvf
import kagefunc as kgf
import vardefunc as vdf
import zzfunc as zzf

from adptvgrnMod import adptvgrnMod
from vsdpir import DPIR
from muvsfunc import SSIM_downsample
from vsutil import *
core = vs.core



###DPIR
def to_rgbs(clip: vs.VideoNode, matrix: int = 1) -> vs.VideoNode:
    clip = depth(clip, 32).std.SetFrameProp('_Matrix', intval=matrix)
    clip = core.resize.Bicubic(clip, format=vs.RGBS)
    return clip


def to_yuvps(clip: vs.VideoNode, matrix: int = 1) -> vs.VideoNode:
    return core.resize.Bicubic(clip, format=vs.YUV420P16, matrix=matrix)



###SCENEFILTER
edstart = 33087

hardsubs_GTFO_ranges = [
    (2734, 3210), (3448, 3525), (22795, 22872),
    (23457, 23507), (24828, 24910)
]



MOVES_LIKE_JAGGED_RANGES = [
    (1794, 1820), (19408, 19465), (24515, 24628),
    (25784, 25897), (26396, 26557), (29860, 29931),
    (32319, 32420)
]



clamp_ranges = [
   (0, 260), (1917, 1964), (2054, 2077),
   (7616, 7663), (7724, 7891), (7966, 8007),
   (9577, 9690), (9781, 9858), (11991, 12233),
   (12354, 13287), (13423, 13500), (14047, 14307),
   (14362, 14466), (14601, 14732), (14958, 15065),
   (16329, 16568), (18508, 19347), (22571, 22642),
   (22690, 22731), (23182, 23226), (23829, 23960),
   (24051, 24194), (24314, 24355), (24629, 24718),
   (24911, 24940), (26882, 27055), (27198, 27239),
   (27692, 27859), (28777, 29106), (31875, 31922),
   (31971, 32123)
]



deband_ranges = [
    (8464, 8622), (18508, 19407)
]



strong_deband_ranges = [
    
]



dpir_ranges = [
    (8707, 9450), (24290, 24313)
]


###SOURCES
waka = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\Sonny Boy\Waka\Sonny Boy E04 [1080p][AAC][JapDub][GerSub][Web-DL].mkv")
waka = depth(waka, 16)
cbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\Sonny Boy\Amazon\Sonny Boy - 04 (Amazon Prime CBR 1080p).mkv")[12:-12]
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
dpir_pass = to_yuvps(DPIR(to_rgbs(aa), strength = 40, task = 'deblock', device_type = 'cpu'))
dark_clip = awf.fixlvls(aa, 0.8)
dpir_pass = core.std.Expr([dark_clip, dpir_pass], "x y max")
dpir_pass = adptvgrnMod(dpir_pass, luma_scaling = 3, strength = 0.40, size = 1.80, sharp = 40, static = False)
dpir_scenefilter = lvf.rfs(aa, dpir_pass, dpir_ranges)

dehalo = gf.MaskedDHA(dpir_scenefilter, rx = 1.6, ry = 1.6, darkstr = 0.0, brightstr = 1.0, showmask = False)
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
#dpir_pass.set_output(9)


#