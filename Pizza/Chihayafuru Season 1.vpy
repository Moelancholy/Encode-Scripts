import vapoursynth as vs
from os import path
from functools import partial

import awsmfunc as awf
import debandshit
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import lvsfunc as lvf
import mvsfunc as mvf
import vardefunc as vdf
import vsutil

from adptvgrnMod import adptvgrnMod
from finedehalo import fine_dehalo
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
src = core.lsmas.LWLibavSource(source=r"E:\Anime\花牌情缘／ちはやふる\CHIHAYAFURU_1\BDMV\STREAM\00003.m2ts")
src = fvf.Depth(src, 16)

shader = r"C:\Users\twint\AppData\Roaming\mpv\shaders\FSRCNNX_x2_56-16-4-1.glsl"
descale_args = dict(height=720, kernel=lvf.kernels.Bicubic(1/3, 1/3))

###SCENEFILTER
ed_ranges = [(30810, 32519)]


###FIX BORDERS (ED has pillarboxing, and needs to be cropped, then bb, then added borders again) (also the scrolling credits of the ED make bbmod go dumdum so no bbmod at the sides FYI)
fb1 = rektlvls(src, rownum=[0,1,2,3,4,1075,1076,1077,1078,1079], rowval=[15,3,-1,2,-1,-1,1,-2,2,13], colnum=[0,1,2,1914,1917,1918,1919], colval=[11,1,-5,-1,-4,5,15])
bb1 = core.edgefixer.ContinuityFixer(fb1, 3, 3, 3, 3, 1080) #bbmod had some troubles that cf hadn't

ed = core.std.CropRel(src, 0, 0, 90, 90)
fb2 = rektlvls(ed, rownum=[0,1,2,3,4,895,896,897,898,899], rowval=[9,-5,6,-2,1,2,-1,6,-3,10], colnum=[0,1,2,3,1914,1917,1918,1919], colval=[12,2,-3,1,-2,-3,5,14])
bb2 = awf.bbmod(fb2, top=5, bottom=5, right=0, left=0, thresh=180, blur=999)
bb2 = core.std.AddBorders(bb2, 0, 0, 90, 90)

bb = lvf.rfs(bb1, bb2, ed_ranges)
bb32 = fvf.Depth(bb, 32)

###RESCALE (NCs don't align with this OP ver, so error mask was used instead of credit mask)
descale = lvf.scale.descale(get_y(bb32), upscaler=None, **descale_args)
descale = fvf.Depth(descale, 16)

descale_m = lvf.scale.descale(get_y(bb32), show_mask=True, **descale_args)
descale_m = vsutil.iterate(descale_m, core.std.Inflate, 25) #yes, really
descale_m = vsutil.iterate(descale_m, core.std.Maximum, 25) #the fucking titles introduce halos out the ass otherwise. this is something someone willing to put more effort than me can fix
descale_m = descale_m.std.Convolution([4]*9)
descale_m = fvf.Depth(descale_m, 16)

dering = haf.HQDeringmod(bb, mrad=3, msmooth=1) #native 1080 elements look hideous so some derining is needed

rescale = vdf.fsrcnnx_upscale(descale, shader_file=shader, profile = 'zastin')
edborder1 = kgf.squaremask(rescale, width=1920, height=5, offset_x=0, offset_y=985)
edborder1 = edborder1.std.Convolution([2]*9)
edborder2 = kgf.squaremask(rescale, width=1920, height=5, offset_x=0, offset_y=90)
edborder2 = edborder2.std.Convolution([2]*9)

merge_scale = core.std.MaskedMerge(rescale, get_y(dering), descale_m)
merge_ed = core.std.MaskedMerge(merge_scale, get_y(bb), edborder1)
merge_ed = core.std.MaskedMerge(merge_ed, get_y(bb), edborder2)
merge_scale = lvf.rfs(merge_scale, merge_ed, ed_ranges)
merge_scale = join([merge_scale, plane(bb, 1), plane(bb,2)])


###DENOISE
ref = haf.SMDegrain(merge_scale, tr=3, prefilter=3, search=3, contrasharp=True, RefineMotion=True, thSAD=150)
dn = mvf.BM3D(merge_scale, ref=ref, sigma=1.1, profile1="fast", radius1=1)


###MASKS
detail_mask = lvf.denoise.detail_mask(dn, brz_a=3000, brz_b=2500)

ed_gray = core.std.ShufflePlanes(dn, 0, vs.GRAY)
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


###DEBAND
altdb = core.placebo.Deband(dn, planes = 1, iterations = 4, threshold = 4.2, radius = 18.0, grain = 1.0, dither = True, dither_algo = 0)
altdb = core.placebo.Deband(altdb, planes = 2|4, iterations = 2, threshold = 4.0, radius = 18.0, grain = 0.0, dither = True, dither_algo = 0)
altdb = core.std.MaskedMerge(altdb, merge_scale, texture_mask)
altdb = core.std.MaskedMerge(altdb, merge_scale, detail_mask)


###OUTPUT
out = adptvgrnMod(altdb, strength=0.2, size=1.00, luma_scaling=8, sharp=70, static=True, lo=19)
final = fvf.Depth(out, 10)
#comp = core.std.Interleave([src, out])
final.set_output()