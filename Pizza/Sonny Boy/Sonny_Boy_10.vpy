import vapoursynth as vs
import debandshit

import awsmfunc as awf
import G41Fun as gf
import havsfunc as haf
import lvsfunc as lvf
import kagefunc as kgf
import vardefunc as vdf
from vsutil.clips import get_y, split
import zzfunc as zzf

from adptvgrnMod import adptvgrnMod
from vsdpir import DPIR
from muvsfunc import SSIM_downsample
from vsutil import *
core = vs.core
core.max_cache_size = 42*1024
core.num_threads = 12

###DPIR
def to_rgbs(clip: vs.VideoNode, matrix: int = 1) -> vs.VideoNode:
    clip = depth(clip, 32).std.SetFrameProp('_Matrix', intval=matrix)
    clip = core.resize.Bicubic(clip, format=vs.RGBS)
    return clip


def to_yuvps(clip: vs.VideoNode, matrix: int = 1) -> vs.VideoNode:
    return core.resize.Bicubic(clip, format=vs.YUV420P16, matrix=matrix)



###SCENEFILTER
edstart = 33088

hardsubs_GTFO_ranges = [
    (2570, 2683), (5291, 5410), (6039, 6110),
    (11550, 11597), (11874, 11939), (15702, 15755),
    (15828, 15893), (17487, 17558), (26148, 26201),
    (26304, 26363), (27102, 27167), (27311, 27364)
]



MOVES_LIKE_JAGGED_RANGES = [
    (2942, 3181), (4106, 4321), (5079, 5168),
    (6039, 6110), (13939, 13983), (31708, 31809),
]



clamp_ranges = [
   (60, 455), (597, 916), (1019, 1099),
   (1460, 2069), (2336, 2362), (2405, 2494), 
   (2570, 2683), (3380, 3619), (3752, 4105),
   (4322, 4487), (4578, 4676), (4935, 5078),
   (5411, 5594), (6111, 6380), (7071, 7226),
   (8190, 8297), (8667, 8729), (8778, 8957),
   (9060, 9146), (9293, 9433), (9530, 9643),
   (10037, 10102), (10241, 10322), (10386, 10676),
   (10679, 11276), (11397, 11471), (11694, 11753),
   (11940, 12032), (12258, 12443), (12594, 12881),
   (12936, 13055), (13097, 13137), (13282, 13395),
   (13467, 13547), (13831, 13938), (13984, 14016),
   (14101, 14295), (14392, 14598), (14683, 15063),
   (15309, 15827), (15894, 16067), (16293, 16334),
   (16452, 16487), (16679, 17138), (17325, 17486),
   (17559, 17708), (17751, 17804), (17856, 18218),
   (18285, 18352), (18492, 18668), (19011, 19333),
   (19375, 19434), (19603, 19860), (20236, 20391),
   (20730, 20849), (20910, 20957), (21030, 21089),
   (21330, 21365), (21437, 21761), (21834, 21929),
   (22050, 22097), (22210, 22442), (22583, 22681),
   (22862, 22959), (23278, 23343), (23434, 23493),
   (23614, 23715), (23902, 23967), (24002, 24093),
   (24202, 24261), (24333, 24620), (25170, 25313),
   (25512, 25703), (25866, 26147), (26304, 26705),
   (26847, 27041), (27102, 27167), (27365, 27772),
   (27875, 27995), (28010, 28213), (28412, 28801),
   (29174, 29887), (30071, 30676), (30785, 30987),
   (31012, 31113), (31246, 31395), (31504, 31707),
   (31810, 31983), (32062, 32127), (32176, 32223),
   (32272, 32565), (32608, 32727), (32776, 32871),
   (32944, 33015), 
]

deband_ranges = [
    (5411, 5521), (6477, 6553)
]



strong_deband_ranges = [
    (6381, 6476), (12033, 12158)
]



dpir_ranges = [
    (4677, 4934), (6554, 6745), (6842, 6923),
    (7227, 7328), (7875, 8024), (8430, 8558),
    (9230, 9292)
]


###SOURCES
waka = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\Sonny Boy\Waka\Sonny Boy E10 [1080p][AAC][JapDub][GerSub][Web-DL].mkv")
waka = depth(waka, 16)
cbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\Sonny Boy\Amazon\Sonny Boy - 10 (Amazon Prime CBR 1080p).mkv")[12:-12]
cbr = depth(cbr, 16).std.Expr("x 128 +")
cbr = cbr[1] + cbr[1:]
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

deband = debandshit.dumb3kdb(preden2, radius=16, threshold=40, grain=30)
deband = core.std.MergeDiff(deband, diff)
deband = core.std.MaskedMerge(deband, dering, deband_mask)

strong_deband = debandshit.dumb3kdb(preden2, radius=16, threshold=64, grain=30)
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

#2942,3181,b=1.4/9746,9847,b=1.3/29840,29977,b=1.3/30314,30552,b=1.2
#19177
#custom 9884, 13138, 15756/16293 (chalk), 22002 (parts of path), 22443 (bubble), 30658 (bubble), 31054 (ground), 31984