import vapoursynth as vs
import ntpath
import os
from os import path
from glob import glob
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
import vsutil
import xvs

from adptvgrnMod import adptvgrnMod
from finedehalo import fine_dehalo
from nnedi3_rpow2 import nnedi3_rpow2
from rektlvl import rektlvls
from vsutil import get_y, join, plane

core = vs.core
core.max_cache_size = 1024 * 64



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
waka = core.lsmas.LWLibavSource(source=r"C:\Users\twint\Desktop\SSSS.Dynazenon E01v2 [1080p][AAC][JapDub][GerSub][Web-DL].mkv")[24:34069]
waka = fvf.Depth(waka, 16)

funi = core.ffms2.Source(source=r"C:\Users\twint\Desktop\ssss_funi_01.mkv")[168:]
funi = fvf.Depth(funi, 16)

shader = r"C:\Users\twint\AppData\Roaming\mpv\shaders\FSRCNNX_x2_56-16-4-1.glsl"
descale_args = dict(height=720, kernel=lvf.kernels.Bicubic(0, 1/2))
deband_args = dict(iterations=4, threshold=3.3, radius=16) #went with aggressive deband bcs cba to scenefilter


###SCENEFILTER
ed_split_ranges = [(31896, 33979)]
hardsubs_GTFO_ranges = [(966, 1019), (9461, 9721), (11222, 11299), (11735, 11788), (16976,17029), (33683, 33950)]


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


###RESCALE 
descale = lvf.scale.descale(get_y(bb32), upscaler=None, **descale_args)
descale = fvf.Depth(descale, 16)
darken = haf.FastLineDarkenMOD(descale, strength=16)

abs_error_m = lvf.scale.descale(get_y(bb32), show_mask=True, **descale_args)
abs_error_m = fvf.Depth(abs_error_m, 16)
abs_error_m = abs_error_m.std.Convolution([4]*9).std.Maximum().std.Inflate()

edborder1 = kgf.squaremask(bb, width=1920, height=5, offset_x=0, offset_y=178)
edborder2 = kgf.squaremask(bb, width=1920, height=5, offset_x=0, offset_y=897)
edborder3 = kgf.squaremask(bb, width=6, height=720, offset_x=957, offset_y=180)
edborder = core.std.Expr([edborder1, edborder2, edborder3], 'x y z max max').std.Convolution([1]*9)
error_ed = core.std.Expr([edborder, abs_error_m], 'x y max')

descale_m = lvf.rfs(abs_error_m, error_ed, ed_split_ranges)
rescale = vdf.fsrcnnx_upscale(darken, shader_file=shader, profile = 'zastin')
#rescale = nnedi3_rpow2(descale).resize.Spline36(1920, 1080)
merge_scale = core.std.MaskedMerge(rescale, get_y(bb), descale_m)
merge_scale = join([merge_scale, plane(bb, 1), plane(bb,2)])


###DENOISE
ref = haf.SMDegrain(merge_scale, tr=2, RefineMotion=True, thSAD=140)
dn = mvf.BM3D(merge_scale, ref=ref, sigma=[1, 0, 0], profile1="fast", radius1=1)
dn = core.knlm.KNLMeansCL(dn, h=0.7, a=4, d=3, s=4, device_type='gpu', device_id=0, channels='UV')
dn = vdf.decsiz(dn, sigmaS=10, min_in=128<<8, max_in=200<<8, gamma=1.1)
dn = haf.ContraSharpening(dn, merge_scale)


###MASKS
ed_gray = core.std.ShufflePlanes(ref, 0, vs.GRAY)
rmask = debandshit.rangemask(ed_gray, 2, 2)
emask = ed_gray.std.Prewitt()

em_hi = emask.std.Binarize(60 * 257, v0=65535, v1=0)
em_hi = vsutil.iterate(em_hi, core.std.Minimum, 5)
em_me = emask.std.Binarize(40 * 257, v0=65535, v1=0)
em_me = vsutil.iterate(em_me, core.std.Minimum, 4)
em_lo = emask.std.Binarize(20 * 257, v0=65535, v1=0)
em_lo = vsutil.iterate(em_lo, core.std.Minimum, 2)
rm_txt = core.std.Expr([rmask, em_hi, em_me, em_lo], 'x y z a min min min')

weighted = LevelsM(rm_txt, points=[x *256 for x in (1.75, 2.5, 5, 10)], levels=[0, 1, 1, 0], xpass=[0,0], return_expr=0)
texture_mask = weighted.std.BoxBlur(hradius=8,vradius=8).std.Expr(f'x {65535 * 0.2} - {1 / (1 - 0.2)} *')

detail_mask = lvf.denoise.detail_mask(ref, brz_a=3000, brz_b=2500)
line_mask1 = kgf.retinex_edgemask(ref)
line_mask2 = line_mask1.std.Binarize(60<<7)
db_mask = core.std.Expr([detail_mask, line_mask2, texture_mask], 'x y z max max') #some extra minor line protection and texture protection
aa_mask = line_mask1.std.Binarize(60<<8).std.Maximum().std.Convolution([1]*9)
cmask = line_mask1.std.Binarize(70<<8).std.Minimum()


###AA
aa = lvf.sraa(dn, rfactor=1.5, rep=13)
aa = core.std.MaskedMerge(dn, aa, aa_mask)

cwarp = xvs.WarpFixChromaBlend(aa, thresh=69)
cwarp = core.std.MaskedMerge(aa, cwarp, cmask)


###DEBAND
db = join([
    core.placebo.Deband(plane(cwarp, 0), grain=1, **deband_args),
    core.placebo.Deband(plane(cwarp, 1), grain=0, **deband_args),
    core.placebo.Deband(plane(cwarp, 2), grain=0, **deband_args)
])
db = core.std.MaskedMerge(db, cwarp, db_mask)


###GRAIN
grain = adptvgrnMod(db, strength=0.2, size=1.00, luma_scaling=8, sharp=80, static=True, lo=19)


###OUTPUT
out = fvf.Depth(grain, 10)
#comp = core.std.Interleave([ref, dn])
out.set_output()