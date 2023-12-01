import vapoursynth as vs

import xvs

import awsmfunc as awf
import havsfunc as haf
import kagefunc as kgf
import lvsfunc as lvf
import vardefunc as vdf

from adptvgrnMod import adptvgrnMod, sizedgrn
import G41Fun as gf
from finedehalo import fine_dehalo
from thresholding import kimagure_thresholding, thresholding, Thresholds
from vsdpir import DPIR
from vsutil import depth, get_y, iterate, plane, split

core = vs.core

###STANDARD IMPORT + CROP + FIX BORDERS
src = core.lsmas.LWLibavSource(r"E:\Anime\KIMAGURE ORANGE ROAD THE MOVIE\BDMV\STREAM\00000.m2ts")
crop = depth(core.std.CropRel(src, 0, 0, 22, 22), 16)
crop_yuv = core.resize.Bicubic(crop, 960, 518, format=vs.YUV444P16)


##FIX BORDERS BBMOD
y, u, v = split(crop)
bb_strong = awf.bbmod(crop, top=2, bottom=2, right=0, left=0, thresh=10, blur=14)

bb_y = awf.bbmod(y, top=2, bottom=2, right=0, left=0, thresh=6, blur=500)
bb_u = awf.bbmod(u, top=1, bottom=1, right=0, left=0, thresh=7, blur=100)
bb_v = awf.bbmod(v, top=1, bottom=1, right=0, left=0, thresh=7, blur=100)

cf_y = core.edgefixer.ContinuityFixer(y, 0, 2, 0, 2, 1036)
cf_u = core.edgefixer.ContinuityFixer(u, 0, 1, 0, 1, 518)
cf_v = core.edgefixer.ContinuityFixer(v, 0, 1, 0, 1, 518)

bb_merge = core.std.ShufflePlanes([bb_y, bb_u, bb_v], [0, 0, 0], colorfamily = vs.YUV)
cf_merge = core.std.ShufflePlanes([cf_y, cf_u, cf_v], [0, 0, 0], colorfamily = vs.YUV)



bb_strong_splice = lvf.rfs(bb_merge, bb_strong,
[
(4923, 4957), (5843, 6002), (8824, 8941), (9096, 9166), (12450, 12521), (12669, 12763), (20330, 20544), (21104, 21258),
(22147, 22343), (21837, 22343), (23062, 23420), (23695, 23802), (24823, 25013), (25151, 25197), (25956, 26494), (27234, 27323),
(29379, 29667), (30385, 30514), (32695, 32837), (33566, 33584), (33655, 33677), (34482, 34594), (34689, 34756), (35312, 35382),
(36222, 36254), (36378, 36441), (36699, 36827), (39319, 39348), (43243, 43361), (46247, 46285), (50748, 50782), (51259, 51313),
(53864, 54102), (56439, 56465), (56593, 56654), (58805, 59043), (62122, 62313), (62648, 62814), (63388, 63449), (64107, 64152),
(64681, 64989), (68220, 68278), (70372, 70496), 
])



cf_splice = lvf.rfs(bb_strong_splice, cf_merge,
[
(1064, 1170), (1788, 1852), (4006, 4165), (8942, 9012), (9368, 9462), (28943, 29122), (29668, 29900), (32039, 32073),
(35699, 35809), (54663, 54729), (60638, 60780), (60924, 61090), 
])



no_bb_splice = lvf.rfs(cf_splice, crop,
[
(1248, 1490), (2456, 2632), (3480, 3569), (4237, 4779), (4958, 5185), (5293, 5675), (6146, 6598), (6867, 7623),
(7875, 8346), (8537, 8823), (9570, 9711), (9995, 10544), (10987, 11069), (11800, 11942), (12190, 12236), (12354, 12449),
(12522, 12668), (13694, 14080), (14116, 14844), (15204, 15250), (15865, 16032), (16822, 17106), (17249, 17323), (18243, 20329),
(20602, 20714), (20838, 21103), (21618, 21836), (22344, 22426), (22642, 22724), (23421, 23599), (23981, 24267), (24603, 24822),
(26569, 27171), (27719, 28115), (28379, 28533), (30515, 30668), (30931, 31847), (32074, 32665), (33291, 33565), (33608, 33654),
(33678, 33724), (33964, 34197), (34339, 34481), (34595, 34617), (34842, 34912), (35050, 35216), (35580, 35698), (35810, 35880),
(35983, 36167), (36255, 36377), (37178, 37511), (37591, 37927), (38085, 38155), (38449, 38555), (40864, 40910), (40804, 41154),
(41213, 41385), (41457, 41821), (41965, 43242), (44397, 44683), (44833, 44939), (45249, 45284), (45452, 45597), (46309, 46379),
(50783, 50853), (51991, 52133), (52588, 52802), (53042, 53114), (53461, 53615), (53693, 53863), (54184, 54238), (54450, 54604),
(54730, 54848), (55264, 55374), (55424, 55638), (58614, 58804), (59044, 59186), (59234, 59318), (60171, 60637), (61091, 61162),
(61306, 61511), (62314, 63245), (63758, 63804), (64243, 64551), (65133, 65303), (65390, 65637), (65924, 66881), (66926, 67008),
(67152, 67434), (67563, 68016), (69013, 69136), (69363, 69649), (69700, 69890), (70128, 70222), (70610, 70926), (71237, 71970),
(72066, 72616), (73012, 73604), (73744, 73820), 
])

#if anyone reading this wants to improve this: separate the luma and chromafixing more thoroughly than here
#that's the main reason it can still be improved (considerably), but only occured to me halfway through
#I'm not fucking restarting this shit though wtf I'm already over 5h of bbmod-peeping in

bb32 = depth(no_bb_splice, 32)
bb32_y = get_y(bb32)

###FIX BORDERS GRAIN
sqmask1 = kgf.squaremask(no_bb_splice, width=1920, height=13, offset_x=0, offset_y=0)
sqmask2 = kgf.squaremask(no_bb_splice, width=1920, height=2, offset_x=0, offset_y=1034)
sqmask = core.std.Expr([sqmask1, sqmask2], "x y max")
sqmaskdown = core.resize.Point(sqmask, width = 960, height = 518).std.Convolution([1]*25).std.Convolution([1]*25).std.Convolution([1]*25)
sqmaskblurred = core.resize.Bicubic(sqmaskdown, 1920, 1036).std.Convolution([1]*25).std.Convolution([1]*25).std.Convolution([1]*25)

bandmask = awf.bandmask(no_bb_splice, 750, pix=12, dec=2, exp=2)
bandmask = core.std.Expr([bandmask, sqmask], "x y min").resize.Point(960, 518)
bandmask = bandmask.std.Convolution([1]*25).std.Convolution([1]*25).std.Convolution([1]*25)
bandmask = core.resize.Bicubic(bandmask, 1920, 1036).std.Convolution([1]*25).std.Convolution([1]*25).std.Convolution([1]*25)

bandgrain1 = sizedgrn(no_bb_splice, strength = 1.00, size = 1.35, sharp = 85, grain_chroma = True)
bandgrain2 = sizedgrn(no_bb_splice, strength = 1.30, size = 1.35, sharp = 85, grain_chroma = True)
bandgrain3 = sizedgrn(no_bb_splice, strength = 1.60, size = 1.35, sharp = 85, grain_chroma = True)

bordenoise = haf.SMDegrain(no_bb_splice, tr=2, RefineMotion=True, thSAD=265, plane = 0)
denoisegrain1 = sizedgrn(bordenoise, strength = 0.85, size = 1.35, sharp = 85, grain_chroma = True)
denoisegrained1 = core.std.MaskedMerge(no_bb_splice, denoisegrain1, sqmaskblurred)

denoisegrain2 = sizedgrn(bordenoise, strength = 1.15, size = 1.35, sharp = 85, grain_chroma = True)
denoisegrained2 = core.std.MaskedMerge(no_bb_splice, denoisegrain2, sqmaskblurred)

denoisegrain3 = sizedgrn(bordenoise, strength = 1.45, size = 1.35, sharp = 85, grain_chroma = True)
denoisegrained3 = core.std.MaskedMerge(no_bb_splice, denoisegrain1, sqmaskblurred)

bordergrained1 = depth(core.std.MaskedMerge(denoisegrained1, bandgrain1, bandmask), 32)
bordergrained2 = depth(core.std.MaskedMerge(denoisegrained2, bandgrain2, bandmask), 32)
bordergrained3 = depth(core.std.MaskedMerge(denoisegrained3, bandgrain3, bandmask), 32)

thrs_a = tuple([0.69, 0.72, 1, 1])
thrs_b = tuple([0.34, 0.42, 0.69, 0.72])
thrs_c = tuple([0.15, 0.25, 0.34, 0.42])
thrs_d = tuple([0.055, 0.14, 0.15, 0.25])

bordergrain_thr = thresholding(
        Thresholds(get_y(bordergrained1), *thrs_a, None, None),
        Thresholds(get_y(bordergrained3), *thrs_b, None, None),
        Thresholds(get_y(bordergrained2), *thrs_c, None, None),
        Thresholds(get_y(bordergrained1), *thrs_d, None, None),
        base = bb32_y
)

bordergrain_thr = depth(core.std.ShufflePlanes([bordergrain_thr, bb32], [0, 1, 2], vs.YUV), 16)


###GENERATING MASKS (EPISODE)
preden = core.knlm.KNLMeansCL(crop_yuv, h=0.5, a=2, d=2, s=2, device_type='gpu', device_id=0, channels = 'YUV').resize.Bicubic(1920, 1036, format=vs.YUV420P16)
preden = core.std.ShufflePlanes([preden, no_bb_splice], [0, 1, 2], vs.YUV)




mask0 = vdf.mask.PrewittStd().get_mask(get_y(preden), multi=2.00).std.Minimum().std.Maximum().std.Maximum()
mask1 = mask0.std.Binarize(40<<7).rgvs.RemoveGrain(20).rgvs.RemoveGrain(20)
mask2 = mask0.std.Binarize(70<<8)
mask3 = mask2.rgvs.RemoveGrain(20).rgvs.RemoveGrain(20)


###DENOISE CHROMA
crop444 = depth(vdf.scale.to_444(bordergrain_thr, 1920, 1036, join_planes = True), 32)
cropRGB = core.resize.Bicubic(crop444, format = vs.RGBS)

cden1 = core.ccd.CCD(cropRGB, 8)
cden2 = core.ccd.CCD(cropRGB, 10)

cden_thr = kimagure_thresholding(cropRGB, cden1, cden2)
cden_thr = core.std.ShufflePlanes([bordergrain_thr, core.resize.Bicubic(cden_thr, format = vs.YUV420P16, matrix = 1)], [0, 1, 2], vs.YUV)

luma_mask = core.adg.Mask(crop.std.PlaneStats(), luma_scaling=15)
cden_scaled = core.std.MaskedMerge(cden_thr, bordergrain_thr, luma_mask)

cden = lvf.rfs(cden_thr, cden_scaled,
[
(9912, 10065), (11070, 11680), (12153, 12189), (12296, 12354), (15490, 15680), (22725, 22799), (26495, 26568), (34339, 34481),
(34618, 34688), (34757, 34841), (35383, 35579), (36408, 37274), (38277, 38448), (41696, 41821), (42108, 43242), (44397, 45105),
(45285, 45451), (52517, 52587), (53209, 53280), (53864, 54102), (55639, 55889), (57097, 57476), (59808, 59950), (60022, 61090),
(61162, 61738), (64061, 64106), (67353, 67487), (67766, 67801), (70223, 70371), (70451, 70609), (77621, 77691), (77809, 77858),
(80267, 80296), (80535, 80992), (81171, 81185), (81767, 81970), (86880, 87085), (87894, 88059), (90098, 90209), (92555, 92613),
(97853, 98051)
])


##FIX CHROMA
#crest1 = xvs.mwcfix(cden_scaled, kernel = 1, restore = 4, a = 2, grad = 4, warp = 3, thresh = 96, blur = 3, repair = 1)
#crest2 = xvs.mwcfix(cden_scaled, kernel = 1, restore = 4, a = 2, grad = 4, warp = 4, thresh = 128, blur = 3, repair = 1)

#crest1 = core.std.MaskedMerge(cden_scaled, crest1, mask1)
#crest = core.std.MaskedMerge(crest1, crest2, mask2)


##DERING
#ring_mask = fine_dehalo(cden_scaled, rx = 2.1, thlimi = 50, thlima = 100, showmask = 1).rgvs.RemoveGrain(20).rgvs.RemoveGrain(20)
#preden2 = core.knlm.KNLMeansCL(core.resize.Bicubic(cden_scaled, 1440, 1080, format=vs.YUV444P16), h=0.5, a=2, d=2, s=2, device_type='gpu', device_id=0, channels = 'YUV').resize.Bicubic(1440, 1080, format=vs.YUV420P16)
#preden2 = core.std.ShufflePlanes([preden2, cden_scaled], [0, 1, 2], vs.YUV)
#preden2 = core.std.MaskedMerge(cden_scaled, preden2, ring_mask)
#diff = core.std.MakeDiff(cden_scaled, preden2)

#dering = fine_dehalo(cden_scaled, rx = 2.1, thlimi = 50, thlima = 100, darkstr = 0, brightstr = 0.88, showmask = 0)
#line_grain = adptvgrnMod(dering, strength=0.50, size=1, luma_scaling=1, static=False, grain_chroma = False)
#dering_m = core.std.MaskedMerge(dering, line_grain, ring_mask)
#dering_merge = core.std.MergeDiff(dering_m, diff)


###OUPUTTING
#grain = adptvgrnMod(dering_merge, strength=0.10, size=1.80, luma_scaling=15, sharp=60, static=False, grain_chroma = False)
#out = depth(grain, 10)
#cut = out[2653:2782] + out[4322:4517] + out[4751:4952] + out[6096:6130] + out[9720:10096] + out[10893:11000] + out[17991:18102] + out[21266:21505] + out[29780:30089]

crop.set_output(1)
bb_merge.set_output(2)
bb_strong.set_output(3)
cf_merge.set_output(4)
cden_scaled.set_output(5)
cden.set_output(6)


#dirtylines scenefilter
#dehalo scenefilter
#potentially dering (hqdering but fine mask?) scenefilter
#potentially chromawarp
#deband scenefilter
#maybe decheckerboard
#zoning

#73898