import vapoursynth as vs
from os import path
from functools import partial

import awsmfunc as awf
import debandshit
import G41Fun as gf
import havsfunc as haf
import kagefunc as kgf
import lvsfunc as lvf
import mvsfunc as mvf
import placebo
import vardefunc as vdf
import zzfunc as zzf

from adptvgrnMod import adptvgrnMod
from rektlvl import rektlvls
from typing import List, Tuple
from vsutil import *

core = vs.core
core.max_cache_size = 1024 * 48
core.num_threads = 8
    


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
src = core.lsmas.LWLibavSource(source=r"E:\Anime\花牌情缘／ちはやふる\CHIHAYAFURU_1\BDMV\STREAM\00003.m2ts")
src = depth(src, 16)

shader = r"C:\Users\twint\AppData\Roaming\mpv\shaders\FSRCNNX_x2_56-16-4-1.glsl"

###SCENEFILTER
op_ranges = []
ed_ranges = [(30810, 32519)]


###FIX BORDERS (ED has pillarboxing, and needs to be cropped, then bb, then added borders again) (also the scrolling credits of the ED make bbmod go dumdum so no bbmod at the sides FYI)
fb1 = rektlvls(src, rownum=[0,1,2,3,4,1075,1076,1077,1078,1079], rowval=[15,3,-1,2,-1,-1,1,-2,2,13], colnum=[0,1,2,1914,1917,1918,1919], colval=[11,1,-5,-1,-4,5,15])
bb1 = core.edgefixer.ContinuityFixer(fb1, 3, 3, 3, 3, 1080) #bbmod had some troubles that cf hadn't

ed = core.std.CropRel(src, 0, 0, 90, 90)
fb2 = rektlvls(ed, rownum=[0,1,2,3,4,895,896,897,898,899], rowval=[9,-5,6,-2,1,2,-1,6,-3,10], colnum=[0,1,2,3,1914,1917,1918,1919], colval=[12,2,-3,1,-2,-3,5,14])
bb2 = awf.bbmod(fb2, top=5, bottom=5, right=0, left=0, thresh=180, blur=999)
bb2 = core.std.AddBorders(bb2, 0, 0, 90, 90)

bb = lvf.rfs(bb1, bb2, ed_ranges)
bb32 = depth(bb, 32)

###RESCALE (NCs don't align with this OP ver, so error mask was used instead of credit mask)
chromadn = core.knlm.KNLMeansCL(bb, h=0.5, a=3, d=2, s=3, device_type='gpu', device_id=0, channels='UV')
descale1 = core.descale.Debicubic(get_y(bb32), 1280, 720, 1/3, 1/3)
descale2 = depth(descale1, 16)
errorscale = core.resize.Bicubic(descale1, 1920, 1080, filter_param_a = 1/3, filter_param_b = 1/3)

ec_ds = haf.EdgeCleaner(descale2, 10, smode=1, hot=True, rmode=13)
rescale = vdf.fsrcnnx_upscale(ec_ds, shader_file=shader, profile = 'zastin', downscaler=None)
rescale = rescale.resize.Bicubic(1920, 1080)


###MASKING RESCALE
abs_error_m = lvf.scale.descale_detail_mask(clip = bb32, rescaled_clip = errorscale).resize.Bicubic(320, 180).std.Inflate().std.Maximum().std.Inflate().std.Maximum().std.Inflate().std.Maximum().std.Inflate().std.Maximum().std.Convolution([4]*9)
abs_error_m = depth(abs_error_m, 16).resize.Bicubic(1920, 1080)
abs_error_m = zzf.combine(zzf.shiftframes(abs_error_m, [-6, 6]))

abs_error_m_ED = zzf.combine(zzf.shiftframes(abs_error_m, [-1, 1])).std.Expr('x 257 *', vs.GRAY16)
edborder1 = kgf.squaremask(rescale, width=1920, height=5, offset_x=0, offset_y=985)
edborder2 = kgf.squaremask(rescale, width=1920, height=5, offset_x=0, offset_y=90)
abs_error_m_ED = core.std.Expr([edborder1, edborder2, abs_error_m_ED], 'x y z max max').std.Convolution([2]*9)

dhmask = haf.HQDeringmod(get_y(bb), show=True)
dh = gf.MaskedDHA(get_y(bb), rx=1.8, ry=1.8, darkstr=0.00, brightstr=0.8, maskpull=35, maskpush=140)
dh = core.std.MaskedMerge(get_y(bb), dh, dhmask)
dering = haf.EdgeCleaner(dh, 12, smode=1, hot=True, rmode=13)

line_m1 = kgf.retinex_edgemask(get_y(bb)).std.Maximum().std.Minimum()
line_m = line_m1.std.Binarize(70<<7).rgvs.RemoveGrain(20).std.Convolution([4]*9)
merge_scale = core.std.MaskedMerge(get_y(bb), rescale, line_m)
merge_scale_dsmasked_ep = core.std.MaskedMerge(rescale, dering, abs_error_m)
merge_scale_dsmasked_ed  = core.std.MaskedMerge(rescale, dering, abs_error_m_ED)

merge_ed = lvf.rfs(merge_scale_dsmasked_ep, merge_scale_dsmasked_ed, ed_ranges)
ec = haf.EdgeCleaner(merge_ed, 6, smode=1, hot=True, rmode=13)
ec = core.std.MaskedMerge(merge_ed, ec, line_m)


###DENOISE
ref = haf.SMDegrain(ec, tr=3, prefilter=3, search=3, RefineMotion=True, thSAD=100)

ed_gray = core.std.ShufflePlanes(ref, 0, vs.GRAY)
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
detail_mask = lvf.mask.detail_mask(bb, brz_a=3000, brz_b=2500)
luma_mask = core.adg.Mask(bb.std.PlaneStats(), luma_scaling=9)

dn_weak = mvf.BM3D(ec, ref=ref, sigma=0.52, profile1="lc", radius1=2)
dn_strong = mvf.BM3D(ec, ref=ref, sigma=0.74, profile1="lc", radius1=2)
dn1 = core.std.MaskedMerge(dn_strong, dn_weak, luma_mask)
dn2 = core.std.MaskedMerge(dn1, ec, line_m1)
decsiz = vdf.decsiz(dn2, sigmaS=4, min_in=208 << 8, max_in=232 << 8)


###DEBAND
merge_final = join([decsiz, plane(chromadn, 1), plane(chromadn, 2)])
dbmask = core.std.Expr([texture_mask, detail_mask, line_m], 'x y z + +')

weakdb = vdf.dumb3kdb(merge_final, radius=16, threshold=[38, 70, 70], grain=0)
strongdb = vdf.dumb3kdb(merge_final, radius=16, threshold=[82, 80, 80], grain=0)
db1 = core.std.MaskedMerge(weakdb, strongdb, luma_mask)
db2 = core.std.MaskedMerge(db1, merge_final, dbmask)
db2 = core.std.MaskedMerge(db2, merge_final, texture_mask)


###OUTPUT
grain = adptvgrnMod(db2, strength=0.11, size=1.20, luma_scaling=4, sharp=75, static=True, lo=19, hi=[200, 240, 240])
grain = adptvgrnMod(db2, strength=0.16, size=1.20, luma_scaling=6, sharp=75, static=False, lo=19, hi=[200, 240, 240])


final = depth(grain, 10)
#comp = core.std.Interleave([dn2, decsiz])
final.set_output()