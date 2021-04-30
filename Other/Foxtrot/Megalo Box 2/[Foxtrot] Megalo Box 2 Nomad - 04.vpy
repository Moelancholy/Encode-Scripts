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
import placebo
import vardefunc as vdf
import xvs
import zzfunc as zzf

from adptvgrnMod import adptvgrnMod
from finedehalo import fine_dehalo
from rektlvl import rektlvls
from vsutil import *

core = vs.core
core.max_cache_size = 1024 * 48



def average_resize(clip, w=None, h=None):
    iw = clip.width
    ih = clip.height
    ow = fallback(w, iw)
    oh = fallback(h, ih)
    
    if (ow > iw and ow/iw != ow//iw) or (oh > ih and oh/ih != oh//ih):
        clip = clip.resize.Point(iw * ceil(ow / iw), ih * ceil(oh / ih))
    
    return clip.fmtc.resample(ow, oh, kernel='box', fulls=1, fulld=1)



def nnedi3_rpow2(clip): return clip.std.Transpose().nnedi3.nnedi3(0,1,0,0,4,2,0,1).std.Transpose().nnedi3.nnedi3(0,1,0,0,4,2,0,1).resize.Spline36(src_left=.5, src_top=.5)



def eedi3s(clip, field=0, alpha=0.125, beta=0.25, gamma=45, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4, mclip=None): 
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
waka = core.lsmas.LWLibavSource(source=r"C:\Users\twint\Desktop\Megalo\04\Megalobox S02E04 [1080p][AAC][JapDub][GerSub][Web-DL].mkv")
waka = waka[0] + waka
waka = depth(waka, 16)

amazon = core.ffms2.Source(source=r"C:\Users\twint\Desktop\Megalo\04\Nomad - Megalo Box 2 - 04 (Amazon Prime CBR 1080p).mkv")
amazon = depth(amazon, 16)


###SCENEFILTER
opstart = 1155 #first frame with any shit thats not black
hardsubs_GTFO_ranges = [(opstart+420, opstart+570)]


###DEHARDSUB
sub_mask_1 = kgf.hardsubmask(waka, amazon)
sub_mask_2 =  core.std.Expr(split(core.std.Expr([waka, amazon], 'x y - abs').resize.Point(format=vs.YUV444P16)), 'x y z max max')
sub_mask_2 = sub_mask_2.std.Binarize(10<<8).std.Minimum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Inflate().std.Inflate().std.Inflate().std.Inflate().std.Inflate()
sub_mask_2 = zzf.combine(zzf.shiftframes(sub_mask_2, [-6, 6])).std.Expr('x 257 *', vs.GRAY16)
sub_mask = lvf.rfs(sub_mask_1, sub_mask_2, hardsubs_GTFO_ranges)

blurred = core.bilateral.Bilateral(get_y(waka), sigmaS=5, sigmaR=0.075)
merge_mask = gf.EdgeDetect(blurred, mode='scharr').rgvs.RemoveGrain(3).std.Binarize(70<<6).std.Convolution([1]*9)
replace_mask = core.std.Expr([merge_mask, sub_mask], 'x y -')
dehardsub1 = core.std.MaskedMerge(get_y(amazon), get_y(waka), replace_mask)
dehardsub2 = core.std.MaskedMerge(waka, amazon, sub_mask)
dehardsub = vdf.merge_chroma(dehardsub1, dehardsub2)


###DIRTYLINES
fb = rektlvls(dehardsub, rownum=[0,1079], rowval=[18,18], colnum=[0,1919], colval=[18,18])
bb = core.edgefixer.ContinuityFixer(fb, 2, 2, 2, 2, 1080)
bb32 = depth(bb, 32)


###DENOISE AND SCALE
chromadn = core.knlm.KNLMeansCL(bb, h=0.55, a=3, d=2, s=3, device_type='gpu', device_id=0, channels='UV')
descale1 = core.descale.Debicubic(get_y(bb32), 1280, 720, 1/3, 1/3)
descale = depth(descale1, 16)
downscale = core.resize.Bicubic(get_y(bb), 1280, 720)
errorscale = descale1.resize.Bilinear(1920, 1080)

line_m =kgf.retinex_edgemask(downscale).std.Binarize(60<<8).std.Minimum()
descale = core.std.MaskedMerge(downscale, descale, line_m)

abs_error_m = lvf.scale.descale_detail_mask(clip = bb32, rescaled_clip = errorscale).resize.Bicubic(320, 180).std.Inflate().std.Maximum().std.Inflate().std.Maximum().std.Inflate().std.Maximum().std.Inflate().std.Maximum().std.Convolution([4]*9)
abs_error_m =depth(abs_error_m, 16).resize.Bicubic(1280, 720)
luma = core.std.MaskedMerge(descale, downscale, abs_error_m)

ed_gray = core.std.ShufflePlanes(luma, 0, vs.GRAY)
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
detail_mask = lvf.mask.detail_mask(luma, brz_a=3000, brz_b=2500)
based_mask = core.std.Expr([detail_mask, texture_mask], 'x y +')

ref = haf.SMDegrain(luma, tr=2, RefineMotion=True, thSAD=120)
dnluma = mvf.BM3D(luma, ref=ref, sigma=0.5, profile1="fast", radius1=2)
dnluma = core.std.MaskedMerge(dnluma, luma, detail_mask)
dnluma = haf.ContraSharpening(dnluma, luma)


###DEHALO
darken = haf.FastLineDarkenMOD(dnluma, strength=12)
dering = haf.EdgeCleaner(darken, 15, smode=1, hot=True)
dhmask = haf.HQDeringmod(dering, show=True)
dh = gf.MaskedDHA(dering, rx=2.0, ry=2.0, darkstr=0.20, brightstr=1.0, maskpull=0, maskpush=60)
dh = core.std.MaskedMerge(dering, dh, dhmask)


###AA
rescale = nnedi3_rpow2(dh)
aawidth, aaheight = 2560, 1440

mask = dh.std.Prewitt().std.Binarize(60 << 8).std.Maximum().rgvs.RemoveGrain(20)
aa_mask = mask.std.Convolution([1]*9)
mclip = average_resize(mask, aawidth, aaheight)
fsr = rescale.std.Transpose()

aa = eedi3s(get_y(fsr), mclip=mclip.std.Transpose())
aa = aa.std.Transpose()
aa = eedi3s(aa, mclip=mclip)
aa = aa.resize.Bicubic(width=1280, height=720)
aa = dh.std.MaskedMerge(aa, aa_mask)
aa = core.rgvs.Repair(aa, dh, 13)


###MERGE
plane_u = core.resize.Bicubic(plane(chromadn, 1), 1280, 720, src_left=.25)
plane_v = core.resize.Bicubic(plane(chromadn, 2), 1280, 720, src_left=.25)
merge = join([aa, plane_u, plane_v])
cwarp = xvs.WarpFixChromaBlend(merge, thresh=80)


###DEBAND
db = vdf.dumb3kdb(cwarp, radius=14, threshold=57, grain=0)
db = core.std.MaskedMerge(db, cwarp, based_mask)


##GRAIN
grain = adptvgrnMod(db, strength=0.12, size=1.00, luma_scaling=10, sharp=60, static=True, lo=19, hi=[200, 240, 240])
grain = adptvgrnMod(grain, strength=0.08, size=1.00, luma_scaling=4, sharp=60, static=False, lo=19, hi=[200, 240, 240])
#upscalecomp = core.resize.Spline36(grain, 1920, 1080)


###OUTPUT
out = depth(grain, 10)
#comp = core.std.Interleave([mergeS, cwarpS])
out.set_output()