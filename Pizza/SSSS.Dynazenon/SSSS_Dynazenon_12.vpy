import vapoursynth as vs
from math import ceil

import awsmfunc as awf
import debandshit
import G41Fun as gf
import havsfunc as haf
import kagefunc as kgf
import lvsfunc as lvf
import mvsfunc as mvf
import placebo
import vardefunc as vdf
import xvs
import zzfunc as zzf

from adptvgrnMod import adptvgrnMod
from rektlvl import rektlvls
from vsutil import *

core = vs.core
core.max_cache_size = 1024 * 32



def average_resize(clip, w=None, h=None):
    iw = clip.width
    ih = clip.height
    ow = fallback(w, iw)
    oh = fallback(h, ih)
    
    if (ow > iw and ow/iw != ow//iw) or (oh > ih and oh/ih != oh//ih):
        clip = clip.resize.Point(iw * ceil(ow / iw), ih * ceil(oh / ih))
    
    return clip.fmtc.resample(ow, oh, kernel='box', fulls=1, fulld=1)



def eedi3s(clip, field=0, alpha=0.125, beta=0.25, gamma=50, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4, mclip=None): 
    out = clip.eedi3m.EEDI3(field, False, 0, alpha, beta, gamma, nrad, mdis, vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, sclip=clip, mclip=mclip)
    if mclip is not None:
        return core.std.Expr([mclip, out, clip], 'x y z ?')
    return out
        


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



###STUFF
waka = core.lsmas.LWLibavSource(source=r"C:\Users\twint\Desktop\SSSS.Dynazenon E12 [1080p][AAC][JapDub][GerSub][Web-DL].mkv")[24:34070]
waka = depth(waka, 16)

amazon = core.ffms2.Source(source=r"C:\Users\twint\Desktop\SSSS.DYNAZENON - 12 (Amazon Prime CBR 1080p).mkv")[24:34070]
amazon = depth(amazon, 16)

shader = r"C:\Users\twint\AppData\Roaming\mpv\shaders\FSRCNNX_x2_56-16-4-1.glsl"
descale_args = dict(height=720, kernel=lvf.kernels.Bicubic(0, 1/2))

weak_deband_args = dict(radius=16, threshold=40, grain=0) 
strong_deband_args = dict(iterations=4, threshold=[4.5, 5.0, 5.0], grain=[1, 0, 0],  radius=18)


###SCENEFILTER
hardsubs_GTFO_ranges = [(25214, 25261), (33686, 34045)]

MOVES_LIKE_JAGGED_RANGES = [(1183, 1428), (4907, 4975), (6043, 6131), (15025, 15050), (23101, 23286)]

no_dn_ranges = [(30612, 30806)]

weak_dn_ranges = []

no_aa_ranges = []

strong_deband_ranges = [(904, 960), (2490, 2549), (2850, 2921), (3318, 3890), (4200, 4397), (4800, 4886), (6213, 6248), (6339, 6485), (8136, 8225), (8545, 8602), (10294, 10431), (13951, 13989)]

chroma_flashback_ranges = [(14038, 14079), (14347, 15024), (15320, 15592), (20768, 20834), (29931, 30611), (30807, 31933), (33263, 33445), (33626, 33757)]


###DEHARDSUB
sub_mask_1 = kgf.hardsubmask(waka, amazon)
sub_mask_2 =  core.std.Expr(split(core.std.Expr([waka, amazon], 'x y - abs').resize.Point(format=vs.YUV444P16)), 'x y z max max')
sub_mask_2 = sub_mask_2.std.Binarize(10<<8).std.Minimum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Inflate().std.Inflate().std.Inflate().std.Inflate().std.Inflate()
sub_mask_2 = zzf.combine(zzf.shiftframes(sub_mask_2, [-6, 6])).std.Expr('x 257 *', vs.GRAY16)
sub_mask = lvf.rfs(sub_mask_1, sub_mask_2, hardsubs_GTFO_ranges)
dehardsub = core.std.MaskedMerge(waka, amazon, sub_mask)


###DIRTYLINES
fb1 = rektlvls(dehardsub, rownum=[0,1079], rowval=[18,18], colnum=[0,1919], colval=[18,18])
bb = core.edgefixer.ContinuityFixer(fb1, 2, 2, 2, 2, 1080)
bb32 = depth(bb, 32)


###DESCALE 
descale1 = core.descale.Debicubic(get_y(bb32), 1280, 720, b=0, c=1/2)
descale = depth(descale1, 16)
errorscale = core.resize.Bicubic(descale1, 1920, 1080, filter_param_a = 0, filter_param_b = 1/2)
darken = haf.FastLineDarkenMOD(descale, strength=16)
darken = get_y(darken)

abs_error_m = lvf.scale.descale_detail_mask(clip = bb32, rescaled_clip = errorscale)
abs_error_m = depth(abs_error_m, 16)
abs_error_m = abs_error_m.std.Convolution([4]*9).std.Maximum().std.Inflate()


###DENOISE
ref = haf.SMDegrain(darken, tr=2, RefineMotion=True, thSAD=150)
dn1 = mvf.BM3D(darken, ref=ref, sigma=0.40, profile1="lc", radius1=2)
dn2 = mvf.BM3D(darken, ref=ref, sigma=0.60, profile1="lc", radius1=2)

dnmask = core.adg.Mask(darken.std.PlaneStats(), luma_scaling=10)
dn = core.std.MaskedMerge(dn2, dn1, dnmask)
dn_weak = core.std.MaskedMerge(dn1, darken, dnmask)
dn = lvf.rfs(dn, dn_weak, weak_dn_ranges)

dn = haf.ContraSharpening(dn, darken)
dn = lvf.rfs(dn, darken, no_dn_ranges)
dnscale = dn.resize.Bicubic(width = 1920, height = 1080)


###RESCALE + AA
rescale = vdf.fsrcnnx_upscale(dn, width = 2560, height = 1440, downscaler = None, shader_file=shader, profile = 'zastin')
downscale = rescale.resize.Bicubic(width = 1920, height=1080)
aawidth, aaheight = 2560, 1440

mask = dnscale.std.Prewitt().std.Binarize(60 << 8).std.Maximum().rgvs.RemoveGrain(20)
aa_mask = mask.std.Convolution([1]*9)
mclip = average_resize(mask, aawidth, aaheight)
fsr = rescale.std.Transpose()

aa1 = lvf.sraa(rescale, rfactor=1, downscaler=None, gamma=40)
aa1 = aa1.resize.Bicubic(width=1920, height=1080)
aa1 = downscale.std.MaskedMerge(aa1, aa_mask)
aa1 = core.rgvs.Repair(aa1, downscale, 13)

aa2 = eedi3s(get_y(fsr), mclip=mclip.std.Transpose())
aa2 = aa2.std.Transpose()
aa2 = eedi3s(aa2, mclip=mclip)
aa2 = aa2.resize.Bicubic(width=1920, height=1080)
aa2 = downscale.std.MaskedMerge(aa2, aa_mask)
aa2 = core.rgvs.Repair(aa2, downscale, 13)

aa = lvf.rfs(aa2, aa1, MOVES_LIKE_JAGGED_RANGES)
aa_dr = haf.EdgeCleaner(aa, 5, smode=1, hot=True, rmode=13)


###MERGE
merge_scale1 = core.std.MaskedMerge(aa_dr, get_y(bb), abs_error_m)
merge_scale = vdf.merge_chroma(merge_scale1, bb)


###MASKS
ed_gray = core.std.ShufflePlanes(merge_scale1, 0, vs.GRAY)
rmask = debandshit.rangemask(ed_gray, 2, 2)
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

detail_mask = lvf.mask.detail_mask(merge_scale1, brz_a=3000, brz_b=2500)
line_mask1 = kgf.retinex_edgemask(merge_scale1)
line_mask2 = line_mask1.std.Binarize(60<<7)
db_mask = core.std.Expr([detail_mask, line_mask2, texture_mask], 'x y z max max')
cmask = line_mask1.std.Binarize(70<<8).std.Minimum()


###CWARP
cwarp = xvs.WarpFixChromaBlend(merge_scale, thresh=60)
cwarp = lvf.rfs(cwarp, merge_scale, chroma_flashback_ranges)


###DEBAND
weakdb = vdf.dumb3kdb(cwarp, **weak_deband_args)
strongdb = placebo.deband(cwarp, **strong_deband_args)
db = lvf.rfs(weakdb, strongdb, strong_deband_ranges)
db = core.std.MaskedMerge(db, cwarp, db_mask)
brighten = haf.FastLineDarkenMOD(db, strength=-14)


###GRAIN
grain = adptvgrnMod(db, strength=0.10, size=1.10, luma_scaling=5, sharp=85, static=True, lo=19, hi=[200, 240, 240])
grain = adptvgrnMod(grain, strength=0.18, size=1.10, luma_scaling=16, sharp=85, static=False, lo=19, hi=[200, 240, 240])


###COMP
#comp1 = awf.FrameInfo(aa2, "clip1")
#comp2 = awf.FrameInfo(aa1, "clip2")
#a = core.std.Interleave([comp1, comp2])

comp3 = awf.FrameInfo(bb, "clip1")
comp4 = awf.FrameInfo(grain, "clip2")
b = core.std.Interleave([comp3, comp4])


###OUTPUT
out = depth(grain, 10)
out.set_output()
