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
    (12764, 12823), (12956, 12991), (16614, 16876),
    (17187, 17222), (17349, 17382)
]



MOVES_LIKE_JAGGED_RANGES = [
    (1467, 1538), (2208, 2294), (2667, 2732),
    (2787, 2816), (2892, 2978), (3348, 3389),
    (3453, 3494), (3684, 3770), (4042, 4112),
    (4257, 4310), (4560, 4961), (5316, 5532),
    (5578, 5685), (5953, 6063), (6289, 6372),
    (6604, 6711), (7388, 7457), (8427, 8739),
    (9066, 9362), (11494, 11601), (12043, 12090),
    (12541, 12667), (12866, 12955), (13039, 13092),
    (13348, 13377), (13630, 13791), (14375, 14601),
    (15848, 16111), (16322, 16510), (17383, 17490),
    (17749, 17895), (18016, 18093), (18442, 18489),
    (18835, 18900), (19204, 19257), (19630, 19767),
    (19876, 19992), (20122, 20373), (20524, 20571),
    (20668, 20691), (20692, 20871), (21385, 21420),
    (23273, 23368), (23813, 23911), (25352, 25429),
    (25667, 25825), (27430, 27498), (28045, 28083),
    (28165, 28212), (28315, 28365), (28456, 28527),
    (28600, 28629), (28834, 28887), (29632, 29690),
    (29994, 30041), (30552, 30647), (31131, 31190),
    (31524, 31589), (32361, 32414)
]



clamp_ranges = [
   (0, 216), (495, 773), (1326, 1373),
   (1860, 1908), (1998, 2087), (2136, 2207),
   (2295, 2342), (3054, 3347), (3495, 3533),
   (4113, 4256), (5686, 5826), (6180, 6243),
   (7000, 7237), (8001, 8222), (9363, 9560),
   (9777, 9944), (10182, 10249), (10310, 10405),
   (10772, 11005), (11138, 11299), (11677, 12042),
   (12091, 12282), (12349, 12540), (12668, 12715),
   (12764, 12865), (12956, 12991), (13093, 13218),
   (13504, 13581), (14674, 15403), (15455, 15791),
   (16112, 16153), (16614, 16928), (17007, 17222),
   (17307, 17382), (17533, 17646), (18490, 18834),
   (18901, 19056), (19093, 19155), (19519, 19629),
   (19768, 19875), (19993, 20052), (20089, 20121),
   (20872, 21168), (21823, 21882), (22204, 22455),
   (22522, 22575), (22784, 22924), (22967, 23074),
   (23231, 23272), (23369, 23428), (23912, 24235),
   (24305, 24388), (25604, 25666), (25898, 26137),
   (26408, 26575), (26840, 26923), (27032, 27121),
   (27331, 27429), (27991, 28044), (28213, 28314),
   (28366, 28455), (29787, 29921), (30282, 30497),
   (30720, 30800), (31245, 31319), (31386, 31523),
   (31590, 32048), (32241, 32360), (32559, 32654),
   (32829, 33086)
]

deband_ranges = [
    (1266, 1325), (6922, 6999), (7833, 7940),
    (10250, 10309), (10310, 10405), (13039, 13218),
    (13378, 13425), (13630, 13737), (17647, 17790),
    (17896, 17943), (17980, 18015), (22087, 22161),
    (24800, 24859), (26576, 26683), (27032, 27079),
    (29440, 29511), (31131, 31190)
]



strong_deband_ranges = [
    (7941, 8000)
]



dpir_ranges = [
    
]


###SOURCES
waka = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\Sonny Boy\Waka\Sonny Boy E09 [1080p][AAC][JapDub][GerSub][Web-DL].mkv")
waka = depth(waka, 16)
cbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\Sonny Boy\Amazon\Sonny Boy - 09 (Amazon Prime CBR 1080p).mkv")[12:-12]
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

#9705,9780,b=1.1/9849,12282,b=1.1/24581,25351,b=1.3/26240,26515.b=1.2/26924,27079,b=1.2/27080,27990,b=1.1/28045,29350,b=1.1/29351,29511,b=1.3/29512,31607,b=1.1/33087,35244,b=0.8
#26924