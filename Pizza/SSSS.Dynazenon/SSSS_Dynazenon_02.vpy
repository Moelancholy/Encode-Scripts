import vapoursynth as vs
from math import ceil
from functools import partial

import awsmfunc as awf
import debandshit
import fvsfunc as fvf
import G41Fun as gf
import havsfunc as haf
import kagefunc as kgf
import lvsfunc as lvf
import mvsfunc as mvf
import vardefunc as vdf
import xvs

from adptvgrnMod import adptvgrnMod
from finedehalo import fine_dehalo
from rektlvl import rektlvls
from vsutil import *

core = vs.core
core.max_cache_size = 1024 * 64



def average_resize(clip, w=None, h=None):
    iw = clip.width
    ih = clip.height
    ow = fallback(w, iw)
    oh = fallback(h, ih)
    
    if (ow > iw and ow/iw != ow//iw) or (oh > ih and oh/ih != oh//ih):
        clip = clip.resize.Point(iw * ceil(ow / iw), ih * ceil(oh / ih))
    
    return clip.fmtc.resample(ow, oh, kernel='box', fulls=1, fulld=1)



def nnedi3_rpow2(clip): return clip.std.Transpose().nnedi3.nnedi3(0,1,0,0,4,2,0,1).std.Transpose().nnedi3.nnedi3(0,1,0,0,4,2,0,1).resize.Spline36(src_left=.5, src_top=.5)



def eedi3s(clip, field=0, alpha=0.125, beta=0.25, gamma=50, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4, mclip=None): 
    out = clip.eedi3m.EEDI3(field, False, 0, alpha, beta, gamma, nrad, mdis, vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, sclip=clip, mclip=mclip)
    if mclip is not None:
        return core.std.Expr([mclip, out, clip], 'x y z ?')
    return out



def eedi3n(clip, field=0, dh=False, alpha=0.2, beta=0.6, gamma=40, nrad=2, mdis=20, vcheck=2, vthresh0=32, vthresh1=64, vthresh2=4, mclip=None): 
    sclip = clip.nnedi3.nnedi3(field,dh,0,0,4,2,0,1)
    out = clip.eedi3m.EEDI3(field, dh, 0, alpha, beta, gamma, nrad, mdis, vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, sclip=sclip, mclip=mclip)
    if mclip is not None:
        if dh:
            mclip = mclip.resize.Point(height=clip.height * 2)
        return core.std.Expr([mclip, out, sclip], 'x y z ?')
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
waka = core.lsmas.LWLibavSource(source=r"C:\Users\twint\Desktop\SSSS.Dynazenon E02 [1080p][AAC][JapDub][GerSub][Web-DL].mkv")[24:34069]
waka = fvf.Depth(waka, 16)

funi = core.ffms2.Source(source=r"C:\Users\twint\Desktop\dnya02funi.mkv")
funi = fvf.Depth(funi, 16)[168:]

shader = r"C:\Users\twint\AppData\Roaming\mpv\shaders\FSRCNNX_x2_56-16-4-1.glsl"
descale_args = dict(height=720, kernel=lvf.kernels.Bicubic(0, 1/2))
deband_args = dict(iterations=3, threshold=3.3, radius=16) #went with aggressive deband bcs cba to scenefilter


###SCENEFILTER
opstart = 1 #first line with emblem
ed_split_ranges = [(31887, 33970)]
hardsubs_GTFO_ranges = [(5593,5599), (10275,10337), (33685,34040)]
MOVES_LIKE_JAGGED_RANGES = [(opstart+29,opstart+290), (opstart+1047,opstart+1077), (opstart+1203,opstart+1428), (17054,17130), (22398,22532), (24566,24610), (29584,29678)]


###DEHARDSUB
sub_mask_1 = kgf.hardsubmask(waka, funi)
sub_mask_2 = kgf.hardsubmask_fades(waka, funi)
sub_mask = lvf.rfs(sub_mask_1, sub_mask_2, hardsubs_GTFO_ranges)
dehardsub = core.std.MaskedMerge(waka, funi, sub_mask)


###DIRTYLINES
fb1 = rektlvls(dehardsub, rownum=[0,1079], rowval=[18,18], colnum=[0,1919], colval=[18,18])
bb1 = core.edgefixer.ContinuityFixer(fb1, 2, 2, 2, 2, 1080)

fb2 = rektlvls(dehardsub, colnum=[959,960], colval=[48,48])
left = core.std.CropRel(fb2, 0, 960, 0, 0)
left =  core.edgefixer.ContinuityFixer(left, 0, 0, 2, 0, 1080)
right = core.std.CropRel(fb2, 960, 0, 0, 0)
right = core.edgefixer.ContinuityFixer(right, 2, 0, 0, 0, 1080)
bb2 = core.std.StackHorizontal([left, right])

bb = lvf.rfs(bb1, bb2, ed_split_ranges)
bb32 = fvf.Depth(bb, 32)


###DESCALE 
descale1 = core.descale.Debicubic(get_y(bb32), 1280, 720, b=0, c=1/2)
descale = fvf.Depth(descale1, 16)
errorscale = core.resize.Bicubic(descale1, 1920, 1080, filter_param_a = 0, filter_param_b = 1/2)
darken = haf.FastLineDarkenMOD(descale, strength=16)

abs_error_m = lvf.scale.descale_detail_mask(clip = bb32, rescaled_clip = errorscale)
abs_error_m = fvf.Depth(abs_error_m, 16)
abs_error_m = abs_error_m.std.Convolution([4]*9).std.Maximum().std.Inflate()

edborder1 = kgf.squaremask(bb, width=1920, height=5, offset_x=0, offset_y=178)
edborder2 = kgf.squaremask(bb, width=1920, height=5, offset_x=0, offset_y=897)
edborder3 = kgf.squaremask(bb, width=6, height=720, offset_x=957, offset_y=180)
edborder = core.std.Expr([edborder1, edborder2, edborder3], 'x y z max max').std.Convolution([1]*9)
error_ed = core.std.Expr([edborder, abs_error_m], 'x y max')
descale_m = lvf.rfs(abs_error_m, error_ed, ed_split_ranges)


###DENOISE
ref = haf.SMDegrain(darken, tr=2, RefineMotion=True, thSAD=150)
dn = mvf.BM3D(get_y(darken), ref=ref, sigma=0.75, profile1="lc", radius1=2)
dn = haf.ContraSharpening(dn, darken)
dnscale = dn.resize.Bicubic(width = 1920, height = 1080)


###RESCALE + AA
rescale = vdf.fsrcnnx_upscale(dn, width = 2560, height = 1440, shader_file=shader, profile = 'zastin')
#rescale = nnedi3_rpow2(descale)
downscale = rescale.resize.Bicubic(width = 1920, height=1080)
aawidth, aaheight = 2560, 1440

mask = dnscale.std.Prewitt().std.Binarize(60 << 8).std.Maximum().rgvs.RemoveGrain(20)
aa_mask = mask.std.Convolution([1]*9)
mclip = average_resize(mask, aawidth, aaheight)
fsr = rescale.std.Transpose()

aa1 = eedi3n(get_y(fsr), mclip=mclip.std.Transpose())
aa1 = aa1.std.Transpose()
aa1 = eedi3n(aa1, mclip=mclip)
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

merge_scale1 = core.std.MaskedMerge(aa, get_y(bb), descale_m)
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

detail_mask = lvf.denoise.detail_mask(merge_scale1, brz_a=3000, brz_b=2500)
line_mask1 = kgf.retinex_edgemask(merge_scale1)
line_mask2 = line_mask1.std.Binarize(60<<7)
db_mask = core.std.Expr([detail_mask, line_mask2, texture_mask], 'x y z max max') #some extra minor line protection and texture protection
cmask = line_mask1.std.Binarize(70<<8).std.Minimum()


###CWARP
cwarp = xvs.WarpFixChromaBlend(merge_scale, thresh=69)
cwarp = core.std.MaskedMerge(merge_scale, cwarp, cmask)


###DEBAND
db = join([
    core.placebo.Deband(plane(cwarp, 0), grain=1, **deband_args),
    core.placebo.Deband(plane(cwarp, 1), grain=0, **deband_args),
    core.placebo.Deband(plane(cwarp, 2), grain=0, **deband_args)
])
db = core.std.MaskedMerge(db, cwarp, db_mask)


###GRAIN
grain = adptvgrnMod(db, strength=0.2, size=1.00, luma_scaling=8, sharp=80, static=True, lo=19, hi=[200, 240, 240])


###OUTPUT
out = fvf.Depth(grain, 10)
#comp = core.std.Interleave([bb, grain])
out.set_output()