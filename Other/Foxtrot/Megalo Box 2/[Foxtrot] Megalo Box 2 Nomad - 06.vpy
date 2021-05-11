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



def eedi3s(clip, field=0, alpha=0.125, beta=0.25, gamma=65, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4, mclip=None): 
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
waka = core.lsmas.LWLibavSource(source=r"C:\Users\twint\Desktop\Megalo\06\Megalobox S02E06 [1080p][AAC][JapDub][GerSub][Web-DL].mkv")
waka = waka[:3716] + waka[3717:]
waka = depth(waka, 16)

amazon = core.ffms2.Source(source=r"C:\Users\twint\Desktop\Megalo\06\Nomad - Megalo Box 2 - 06 (Amazon Prime CBR 1080p).mkv")[1:]
amazon = depth(amazon, 16)

shader = r"C:\Users\twint\AppData\Roaming\mpv\shaders\FSRCNNX_x2_56-16-4-1.glsl"


###SCENEFILTER
opstart = 1560 #first frame with any shit thats not black
edstart = 32486 #first frame after hasta ver la luz
edranges = [(edstart, edstart+2056)]
hardsubs_GTFO_ranges = [(opstart+420, opstart+569), (11982, 12052), (13404, 13454), (28230, 28283)]


###DEHARDSUB
sub_mask_1 = kgf.hardsubmask(waka, amazon)
sub_mask_2 =  core.std.Expr(split(core.std.Expr([waka, amazon], 'x y - abs').resize.Point(format=vs.YUV444P16)), 'x y z max max')
sub_mask_2 = sub_mask_2.std.Binarize(10<<8).std.Minimum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Inflate().std.Inflate().std.Inflate().std.Inflate().std.Inflate()
sub_mask_2 = zzf.combine(zzf.shiftframes(sub_mask_2, [-6, 6])).std.Expr('x 257 *', vs.GRAY16)
sub_mask = lvf.rfs(sub_mask_1, sub_mask_2, hardsubs_GTFO_ranges)
dehardsub2 = core.std.MaskedMerge(waka, amazon, sub_mask)
dehardsub = vdf.merge_chroma(get_y(amazon), dehardsub2)


###DIRTYLINES
fb = rektlvls(dehardsub, rownum=[0,1079], rowval=[18,18], colnum=[0,1919], colval=[18,18])
bb = core.edgefixer.ContinuityFixer(fb, 2, 2, 2, 2, 1080)
bb32 = depth(bb, 32)


###SCALE
chromadn = core.knlm.KNLMeansCL(bb, h=0.55, a=3, d=2, s=3, device_type='gpu', device_id=0, channels='UV')
descale1 = core.descale.Debicubic(get_y(bb32), 1280, 720, 1/3, 1/3)
descale2 = depth(descale1, 16)

sd_descale = lvf.scale.descale(descale1, upscaler=None, height=406, kernel=lvf.kernels.Spline16())
sd_descale = depth(sd_descale, 16)
sd_rescale1 = vdf.fsrcnnx_upscale(sd_descale, height = sd_descale.height *2, downscaler=None, shader_file=shader, profile = 'zastin')
sd_upscale = sd_descale.resize.Bicubic(sd_rescale1.width, sd_rescale1.height)
merge_mask = sd_rescale1.std.Prewitt().std.Binarize(70 << 6).std.Maximum().rgvs.RemoveGrain(20)
sd_rescale = core.std.MaskedMerge(sd_rescale1, core.std.Merge(sd_rescale1, sd_upscale, 0.5), merge_mask)
sd_rescale2 = core.resize.Bicubic(sd_rescale, 1280, 720)


###AA
rescale = nnedi3_rpow2(sd_rescale)
rescale = rescale.resize.Bicubic(2560, 1440)
mask = sd_rescale2.std.Prewitt().std.Binarize(60 << 8).std.Maximum().rgvs.RemoveGrain(20)
aa_mask = mask.std.Convolution([1]*9)
mclip = average_resize(mask, rescale.width, rescale.height)
fsr = rescale.std.Transpose()

aa = eedi3s(get_y(fsr), mclip=mclip.std.Transpose())
aa = aa.std.Transpose()
aa = eedi3s(aa, mclip=mclip)
aa = aa.resize.Bicubic(width=1280, height=720)

aa = sd_rescale2.std.MaskedMerge(aa, aa_mask)
aa = core.rgvs.Repair(aa, sd_rescale2, 13)
sd_replace = lvf.rfs(aa, descale2, edranges)


###MASK 1080 ELEMENTS
downscale = core.resize.Bicubic(get_y(bb), 1280, 720)
errorscale = descale1.resize.Bicubic(1920, 1080)
abs_error_m = lvf.scale.descale_detail_mask(clip = bb32, rescaled_clip = errorscale).resize.Bicubic(320, 180).std.Inflate().std.Maximum().std.Inflate().std.Maximum().std.Inflate().std.Maximum().std.Inflate().std.Maximum().std.Convolution([4]*9)
abs_error_m = depth(abs_error_m, 16).resize.Bicubic(1280, 720)
luma = core.std.MaskedMerge(sd_replace, downscale, abs_error_m)


###MASKS
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


###DEHALO
dering = haf.EdgeCleaner(luma, 10, smode=1, hot=True)
dhmask = haf.HQDeringmod(dering, show=True)
dh = gf.MaskedDHA(dering, rx=2.0, ry=2.0, darkstr=0.00, brightstr=1.0, maskpull=0, maskpush=60)
dh = core.std.MaskedMerge(dering, dh, dhmask)


###MERGE
plane_u = core.resize.Bicubic(plane(chromadn, 1), 1280, 720, src_left=.25)
plane_v = core.resize.Bicubic(plane(chromadn, 2), 1280, 720, src_left=.25)
merge = join([dh, plane_u, plane_v])
cwarp = xvs.WarpFixChromaBlend(merge, thresh=80)


###DEBAND
db = vdf.dumb3kdb(cwarp, radius=18, threshold=55, grain=2)
db = core.std.MaskedMerge(db, cwarp, based_mask)


##GRAIN
grain = adptvgrnMod(db, strength=0.12, size=1.20, luma_scaling=12, sharp=80, static=True, lo=19, hi=[200, 240, 240])
grain = adptvgrnMod(grain, strength=0.07, size=1.20, luma_scaling=4, sharp=80, static=False, lo=19, hi=[200, 240, 240])


###COMP
#flt = core.resize.Spline36(grain, width=1920, height=1080)
#org = core.resize.Bicubic(dehardsub, format=flt.format)


###OUTPUT
out = depth(grain, 10)
#comp = core.std.Interleave([org, flt])
out.set_output()