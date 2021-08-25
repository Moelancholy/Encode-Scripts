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




###STUFF
def LevelsM(clip, points, levels, xpass=[0, 'peak'], return_expr=False):
    qm = len(points)
    peak = [(1 << clip.format.bits_per_sample) - 1, 1][clip.format.sample_type]
   
    if len(set(xpass)) == 1:
        expr = f'x {points[0]} < x {points[-1]} > or {xpass[0]} '
        qm -= 1
    else:
        expr = f'x {points[0]} < {xpass[0]} x {points[-1]} > {xpass[-1]} '
   
    for x in range(len(points) - 1):
        if points[x+1] < points[-1]:
            expr += f' x {points[x+1]} <= '
        if levels[x] == levels[x+1]:
            expr += f' {peak * levels[x]} '
        else:
            expr += f' x {points[x]} - {peak * (levels[x+1] - levels[x])/(points[x+1] - points[x])} * {peak * levels[x]} + '
   
    for _ in range(qm):
        expr += ' ? '
   
    expr = expr.replace('  ', ' ').replace('peak', f'{peak}')
   
    if return_expr:
        return expr
   
    return clip.std.Expr(expr)



def maxm(clip: vs.VideoNode, sy: int = 2, sc: int = 2) -> vs.VideoNode:
    yp = sy>=sc
    yiter = 1 if yp else 0
    cp = sc>=sy
    citer = 1 if cp else 0
    planes = [0] if yp and not cp else [1,2] if cp and not yp else [0,1,2]
    coor = [0, 1, 0, 1, 1, 0, 1, 0] if (max(sy,sc) % 3) != 1 else [1] * 8

    if sy>0 or sc>0:
        return maxm(clip.std.Maximum(planes=planes, coordinates=coor), sy=sy-yiter, sc=sc-citer)
    else:
        return clip



def minm(clip: vs.VideoNode, sy: int = 2, sc: int = 2) -> vs.VideoNode:
    yp = sy>=sc
    yiter = 1 if yp else 0
    cp = sc>=sy
    citer = 1 if cp else 0
    planes = [0] if yp and not cp else [1,2] if cp and not yp else [0,1,2]
    coor = [0, 1, 0, 1, 1, 0, 1, 0] if (max(sy,sc) % 3) != 1 else [1] * 8

    if sy>0 or sc>0:
        return minm(clip.std.Minimum(planes=planes, coordinates=coor), sy=sy-yiter, sc=sc-citer)
    else:
        return clip



def rangemask(clip: vs.VideoNode, rad: int = 2, radc: Optional[int] = None) -> vs.VideoNode:
    isRGB = clip.format.color_family == vs.RGB
    radc = (rad if isRGB else 0) if radc is None else radc
    if radc == 0:
        clip = togray(clip, int16=False)
    ma = maxm(clip, rad, radc)
    mi = minm(clip, rad, radc)

    expr = 'x y -'
    if not rad:
        expr = ['', 'x y -']
    if not radc:
        expr = ['x y -', '']

    return core.std.Expr([ma, mi], expr)



###SCENEFILTER
hardsubs_GTFO_ranges = [
    (240, 281), (7217, 7252), (7348, 7401), 
    (7447, 7503), (13172, 13303), (19263, 19622), 
    (23923, 23970), (24163, 24228)
]



MOVES_LIKE_JAGGED_RANGES = [

]



deband_ranges = [

]


###SOURCES
waka = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\Sonny Boy\Waka\Sonny Boy E01 [1080p][AAC][JapDub][GerSub][Web-DL].mkv")
waka = depth(waka, 16)
cbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\Sonny Boy\Amazon\Sonny Boy - 01 (Amazon Prime CBR 1080p).mkv")[12:-12]
cbr = depth(cbr, 16).std.Expr("x 128 +")
waka = waka[1] + waka[:17183] + cbr[17184:17216] + waka[17183:]

shader = r"C:\Users\twint\AppData\Roaming\mpv\shaders\FSRCNNX_x2_56-16-4-1.glsl"



###DEHARDSUB
sub_mask_1 = kgf.hardsubmask(waka, cbr)
sub_mask_2 =  core.std.Expr(split(core.std.Expr([waka, cbr], 'x y - abs').resize.Point(format=vs.YUV444P16)), 'x y z max max')
sub_mask_2 = sub_mask_2.std.Binarize(10<<8).std.Minimum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Inflate().std.Inflate().std.Inflate().std.Inflate().std.Inflate()
sub_mask_2 = zzf.combine(zzf.shiftframes(sub_mask_2, [-6, 6])).std.Expr('x 257 *', vs.GRAY16)
sub_mask = lvf.rfs(sub_mask_1, sub_mask_2, hardsubs_GTFO_ranges)
dehardsub = core.std.MaskedMerge(waka, cbr, sub_mask)
preden = haf.SMDegrain(get_y(dehardsub), tr=2, RefineMotion=True, thSAD=150)



###MASKS
ed_gray = core.std.ShufflePlanes(preden, 0, vs.GRAY)
rmask = rangemask(ed_gray, 2, 2)
emask = ed_gray.std.Prewitt()
em_hi = emask.std.Binarize(60 * 257, v0=65535, v1=0)
em_hi = iterate(em_hi, core.std.Minimum, 5)
em_me = emask.std.Binarize(40 * 257, v0=65535, v1=0)
em_me = iterate(em_me, core.std.Minimum, 4)
em_lo = emask.std.Binarize(20 * 257, v0=65535, v1=0)
em_lo = iterate(em_lo, core.std.Minimum, 2)
rm_txt = core.std.Expr([rmask, em_hi, em_me, em_lo], 'x y z a min min min')
weighted = LevelsM(rm_txt, points=[x *256 for x in (1.75, 2.5, 5, 10)], levels=[0, 1, 1, 0], xpass=[0,0], return_expr=0)
texture_mask = weighted.std.BoxBlur(hradius=8,vradius=8).std.Expr(f'x {65535 * 0.2} - {1 / (1 - 0.2)} *')
texture_mask2 = texture_mask.std.Binarize(90<<8)



###AA
upscale =  vdf.scale.fsrcnnx_upscale(get_y(dehardsub), width = 3840, height = 2160, downscaler = None, shader_file=shader, profile = 'zastin')
downscale = depth(SSIM_downsample(upscale, w = 2560, h = 1440, sigmoid = True), 16)
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

aa_y = lvf.rfs(aa1, aa2, MOVES_LIKE_JAGGED_RANGES)
aa = core.std.ShufflePlanes([aa_y, dehardsub], [0, 1, 2], vs.YUV)



###DERING
dehalo = gf.MaskedDHA(aa, rx = 1.6, ry = 1.6, darkstr = 0.0, brightstr = 1.0, showmask = False)
dering = haf.EdgeCleaner(dehalo, strength = 6, rep = True, rmode = 13, smode = 1, hot = True)



###DEBAND
preden2 = haf.SMDegrain(dering, tr=2, RefineMotion=True, thSAD=150)
deband_mask = kgf.retinex_edgemask(preden2)
diff = core.std.MakeDiff(dering, preden2)
deband = debandshit.dumb3kdb(preden2, radius=16, threshold=48, grain=0)
deband = core.std.MergeDiff(deband, diff)
deband = core.std.MaskedMerge(deband, dering, deband_mask)
flt = lvf.rfs(dering, deband, deband_ranges)


###OUTPUT
out = depth(flt, 10)
dehardsub.set_output(0)
flt.set_output(1)



#3.21
#8.04