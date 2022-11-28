import vapoursynth as vs
core = vs.core

core.max_cache_size = 40*1024
core.num_threads = 12

import awsmfunc as awf
import debandshit as dbs
import lvsfunc as lvf
import vardefunc.mask as mask

from adptvgrnMod import adptvgrnMod
from havsfunc import FineDehalo, SMDegrain, HQDeringmod
from kagefunc import retinex_edgemask
from lvsfunc.recon import chroma_reconstruct
from vardefunc.scale import to_444
from vsdpir import DPIR
from vsutil import get_depth, get_y, depth, plane, iterate



###CUSTOM FUNCS
def eedi3s(clip, field=0, alpha=0.125, beta=0.25, gamma=50, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4, mclip=None): 
    out = clip.eedi3m.EEDI3(field, False, 0, alpha, beta, gamma, nrad, mdis, vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, sclip=clip, mclip=mclip)
    if mclip is not None:
        return core.std.Expr([mclip, out, clip], 'x y z ?')
    return out


def merge_tvcr(tv, cr, radius=3, passes=2, lehmer=True):
    tv_high = core.std.MakeDiff(tv, tv.std.BoxBlur(hradius=radius, hpasses=passes, vradius=radius, vpasses=passes))
    tv_low = core.std.MakeDiff(tv, tv_high)
    cr_high = core.std.MakeDiff(cr, cr.std.BoxBlur(hradius=radius, hpasses=passes, vradius=radius, vpasses=passes))
    if lehmer:
        cr_hi = core.std.Expr([tv_high, cr_high], "x 32768 - 3 pow y 32768 - 3 pow + x 32768 - 2 pow y 32768 - 2 pow 0.0001 + + / 32768 +")
    return core.std.MergeDiff(tv_low, cr_hi)


def ccdmod(src: vs.VideoNode, threshold: float = 4, matrix: int = 1) -> vs.VideoNode:
    """
    A faster implementation that processes only chroma. Taken from jvsfunc.
    """

    if src.format.color_family != vs.YUV:
        raise ValueError('ccdmod: only YUV format is supported, use jvsfunc.ccd() for RGB.')

    if matrix is None:
        matrix = 1 if src.width > 1270 or src.height > 710 else 6

    thr = threshold**2/195075.0
    u = plane(src, 1)

    yuv = src.resize.Bicubic(u.width, u.height, format=vs.YUV444P16)
    rgbs = yuv.resize.Point(format=vs.RGBS, matrix_in=matrix)

    rrr = core.std.ShufflePlanes([rgbs, rgbs, rgbs], [0, 0, 0], vs.RGB)
    ggg = core.std.ShufflePlanes([rgbs, rgbs, rgbs], [1, 1, 1], vs.RGB)
    bbb = core.std.ShufflePlanes([rgbs, rgbs, rgbs], [2, 2, 2], vs.RGB)

    ex_ccd = core.akarin.Expr([yuv, rrr, ggg, bbb],
                              ['',
                               'y[-12,-12] y - 2 pow z[-12,-12] z - 2 pow a[-12,-12] a - 2 pow + + A! '
                               'y[-4,-12] y - 2 pow z[-4,-12] z - 2 pow a[-4,-12] a - 2 pow + + B! '
                               'y[4,-12] y - 2 pow z[4,-12] z - 2 pow a[4,-12] a - 2 pow + + C! '
                               'y[12,-12] y - 2 pow z[12,-12] z - 2 pow a[12,-12] a - 2 pow + + D! '
                               'y[-12,-4] y - 2 pow z[-12,-4] z - 2 pow a[-12,-4] a - 2 pow + + E! '
                               'y[-4,-4] y - 2 pow z[-4,-4] z - 2 pow a[-4,-4] a - 2 pow + + F! '
                               'y[4,-4] y - 2 pow z[4,-4] z - 2 pow a[4,-4] a - 2 pow + + G! '
                               'y[12,-4] y - 2 pow z[12,-4] z - 2 pow a[12,-4] a - 2 pow + + H! '
                               'y[-12,4] y - 2 pow z[-12,4] z - 2 pow a[-12,4] a - 2 pow + + I! '
                               'y[-4,4] y - 2 pow z[-4,4] z - 2 pow a[-4,4] a - 2 pow + + J! '
                               'y[4,4] y - 2 pow z[4,4] z - 2 pow a[4,4] a - 2 pow + + K! '
                               'y[12,4] y - 2 pow z[12,4] z - 2 pow a[12,4] a - 2 pow + + L! '
                               'y[-12,12] y - 2 pow z[-12,12] z - 2 pow a[-12,12] a - 2 pow + + M! '
                               'y[-4,12] y - 2 pow z[-4,12] z - 2 pow a[-4,12] a - 2 pow + + N! '
                               'y[4,12] y - 2 pow z[4,12] z - 2 pow a[4,12] a - 2 pow + + O! '
                               'y[12,12] y - 2 pow z[12,12] z - 2 pow a[12,12] a - 2 pow + + P! '
                               f'A@ {thr} < 1 0 ? B@ {thr} < 1 0 ? C@ {thr} < 1 0 ? D@ {thr} < 1 0 ? '
                               f'E@ {thr} < 1 0 ? F@ {thr} < 1 0 ? G@ {thr} < 1 0 ? H@ {thr} < 1 0 ? '
                               f'I@ {thr} < 1 0 ? J@ {thr} < 1 0 ? K@ {thr} < 1 0 ? L@ {thr} < 1 0 ? '
                               f'M@ {thr} < 1 0 ? N@ {thr} < 1 0 ? O@ {thr} < 1 0 ? P@ {thr} < 1 0 ? '
                               '+ + + + + + + + + + + + + + + 1 + Q! '
                               f'A@ {thr} < x[-12,-12] 0 ? B@ {thr} < x[-4,-12] 0 ? '
                               f'C@ {thr} < x[4,-12] 0 ? D@ {thr} < x[12,-12] 0 ? '
                               f'E@ {thr} < x[-12,-4] 0 ? F@ {thr} < x[-4,-4] 0 ? '
                               f'G@ {thr} < x[4,-4] 0 ? H@ {thr} < x[12,-4] 0 ? '
                               f'I@ {thr} < x[-12,4] 0 ? J@ {thr} < x[-4,4] 0 ? '
                               f'K@ {thr} < x[4,4] 0 ? L@ {thr} < x[12,4] 0 ? '
                               f'M@ {thr} < x[-12,12] 0 ? N@ {thr} < x[-4,12] 0 ? '
                               f'O@ {thr} < x[4,12] 0 ? P@ {thr} < x[12,12] 0 ? '
                               '+ + + + + + + + + + + + + + + x + Q@ /'])

    ex_ccd = ex_ccd.resize.Bicubic(src_left=-0.25)
    ex_ccd = depth(ex_ccd, get_depth(src))
    return core.std.ShufflePlanes([src, ex_ccd], [0, 1, 2], vs.YUV)


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


def texture_mask(clip, range):
    ed_gray = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    rmask = mask.MinMax(range).get_mask(clip, lthr=0, multi=1.00)
    emask = ed_gray.std.Prewitt()
    em_hi = emask.std.Binarize(60 * 257, v0=65535, v1=0)
    em_hi = iterate(em_hi, core.std.Minimum, 5)
    em_me = emask.std.Binarize(40 * 257, v0=65535, v1=0)
    em_me = iterate(em_me, core.std.Minimum, 4)
    em_lo = emask.std.Binarize(20 * 257, v0=65535, v1=0)
    em_lo = iterate(em_lo, core.std.Minimum, 2)
    rm_txt = core.std.Expr([rmask, em_hi, em_me, em_lo], 'x y z a min min min')
    weighted = LevelsM(rm_txt, points=[x *256 for x in (1.75, 2.5, 5, 10)], levels=[0, 1, 1, 0], xpass=[0,0], return_expr=0)

    masked = weighted.std.BoxBlur(hradius=8,vradius=8).std.Expr(f'x {65535 * 0.2} - {1 / (1 - 0.2)} *')
    return masked


###SCENEFILTER
MOVES_LIKE_JAGGED_RANGES = [

]

dering_ranges = [
    
]


cr_signs_ranges = [
    
]


ed_ranges = [
    (32368, 34524)
]


###SOURCES
cbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Mobile Suit Gundam - The Witch from Mercury - 06 (Amazon dAnime CBR 1080p).mkv")
vbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Mobile Suit Gundam - The Witch from Mercury - 06 (Amazon dAnime VBR 1080p).mkv")
disney = depth(core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Mobile Suit Gundam - The Witch from Mercury - 03 (Disney+ 1080p).mkv"), 16)[24:]
cr = depth(core.lsmas.LWLibavSource(r"E:\Anime\[SubsPlease] Mobile Suit Gundam - The Witch from Mercury - 06 (1080p) [AFC815D7].mkv"), 16)

blank = core.std.BlankClip(disney)
disney = blank[:192] + disney
cr = blank[:192] + cr

src = lvf.rfs(cr, disney, cr_signs_ranges)

cbr16 = depth(cbr, 16)
ed = cbr16[:192] + src[192:]


##FIX BORDERS
#borders1 = rektlvls(ed, colnum = [0, 1919], colval = [43, 43])
borders2 = awf.bbmod(ed, left = 1, right = 1, thresh = 5, blur = 999, u = True, v = True)
borders2 = lvf.rfs(borders2, cr, cr_signs_ranges)
lehmer = merge_tvcr(vbr, cbr)

###DEBLOCK
deblock1 = depth(to_444(lehmer, 1920, 1080, join_planes = True), 32)
deblock1 = core.resize.Bicubic(deblock1, format = vs.RGBS)
deblock1 = DPIR(deblock1, strength = 6, task = 'deblock', device_type = 'cpu', fp16 = False, tile_x = 540, tile_y = 270, tile_pad=10)
deblock1 = core.resize.Bicubic(deblock1, format = vs.YUV420P16, matrix = 1)

deblock2 = depth(to_444(lehmer, 1920, 1080, join_planes = True), 32)
deblock2 = core.resize.Bicubic(deblock2, format = vs.RGBS)
deblock2 = DPIR(deblock2, strength = 6, task = 'deblock', device_type = 'cpu', fp16 = False, tile_x = 540, tile_y = 270, tile_pad=10)
deblock2 = core.resize.Bicubic(deblock2, format = vs.YUV420P16, matrix = 1)
deblock2 = adptvgrnMod(deblock2, strength = 0.30, static = True, luma_scaling = 2, grain_chroma = False, size = 1.2, sharp = 80)
deblock2 = FineDehalo(deblock2, rx = 2.0, darkstr = 0, brightstr = 0.7)


##AA
borders_y = get_y(borders2)
aa_mask = borders_y.std.Prewitt().std.Binarize(60 << 8).std.Maximum().std.Convolution([1]*9)

aastrong = lvf.aa.upscaled_sraa(borders_y)
aaweak = lvf.aa.taa(borders_y, lvf.aa.nnedi3(opencl=True))
aa1 = lvf.aa.clamp_aa(borders_y, aaweak, aastrong, 1.5)
aa = lvf.rfs(aa1, aastrong, MOVES_LIKE_JAGGED_RANGES)

aa_masked = core.std.MaskedMerge(borders_y, aa, aa_mask)


###DEHALO
double = aa_masked.znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3).std.Transpose().znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3).std.Transpose()
dehalo1 = FineDehalo(double, rx = 2, darkstr = 0, brightstr = 0.9).resize.Point(1920, 1080)
dehalo2 = FineDehalo(aa_masked, rx = 1.7, darkstr = 0, brightstr = 0.7)
dehalo3 = core.std.Merge(dehalo1, dehalo2)
dh32 = depth(dehalo3, 32)


###DENOISE LUMA
preden = depth(SMDegrain(dehalo3, tr = 2, thSAD = 160, RefineMotion = True, prefilter = 2), 32)
denoise_y = core.bm3dcuda.BM3D(clip = dh32, ref = preden, sigma = 0.80, radius = 2).bm3d.VAggregate(radius = 2)
denoise_y = core.bm3dcuda.BM3D(clip = dh32, ref = depth(denoise_y, 32), sigma = 0.80, radius = 2).bm3d.VAggregate(radius = 2)


###DERING
merge1 = core.std.ShufflePlanes([denoise_y, borders2], [0, 1, 2], vs.YUV)
merge1 = deblock1[:192] + merge1[192:]
dering1 = HQDeringmod(merge1, mrad = 3, msmooth = 1, mthr = 20, nrmode = 1)
dering = lvf.rfs(merge1, dering1, dering_ranges)


###CHROMA
chromadn = ccdmod(dering, 4)
linemask = retinex_edgemask(aa_masked)
dn_merged = core.std.MaskedMerge(chromadn, dering, linemask)

luma = get_y(dn_merged)
test1 = chroma_reconstruct(dn_merged, i444 = True)
test2 = test1.resize.Bicubic(format = vs.RGBS)
test2 = DPIR(test2, strength = 10, task = 'deblock', device_type = 'cpu').resize.Bicubic(format = vs.YUV444P16, matrix = 1)
test2 = core.std.ShufflePlanes([dn_merged, test2], [0, 1, 2], vs.YUV)

u_recon = plane(test1, 1)
v_recon = plane(test1, 2)

v_dh1 = FineDehalo(v_recon, rx = 1.7, darkstr=0, brightstr=1)
v_double = v_recon.znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3).std.Transpose().znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3).std.Transpose()
v_dh2 = FineDehalo(v_double, rx = 2.0, darkstr=0, brightstr=1).resize.Point(1920, 1080)
v_dh = core.std.Expr([v_dh1, v_dh2], "x y min")

test3 = core.std.ShufflePlanes([dn_merged, u_recon, v_dh], [0, 0, 0], vs.YUV)
chromafix = core.resize.Bicubic(test3, format = vs.YUV420P16)


###DEBAND
deband = dbs.dumb3kdb(chromafix, 17, [28, 36, 36], 0)
deband = adptvgrnMod(deband, strength = 0.04, size = 1.2, sharp = 80, luma_scaling = 2, static = True, grain_chroma = False)
deband = adptvgrnMod(deband, strength = 0.08, size = 1.2, sharp = 80, luma_scaling = 8, static = False, grain_chroma = False)

strongdeband = dbs.dumb3kdb(chromafix, 17, [48, 50, 50], 0)
strongdeband = adptvgrnMod(deband, strength = 0.05, size = 1.2, sharp = 80, luma_scaling = 2, static = True, grain_chroma = False)
strongdeband = adptvgrnMod(deband, strength = 0.08, size = 1.2, sharp = 80, luma_scaling = 8, static = False, grain_chroma = False)

tmask = texture_mask(get_y(chromafix), 1)
debandmask = core.std.Expr([tmask, linemask], "x y max")
debanded = lvf.rfs(deband, strongdeband, cr_signs_ranges)
deband_masked = core.std.MaskedMerge(debanded, chromafix, debandmask)


###OUTPUT
ed_splice = lvf.rfs(deband_masked, ed, ed_ranges)
grain = adptvgrnMod(deband_masked, strength = 0.02, static = True, luma_scaling = 2, grain_chroma = False, size = 1.2, sharp = 80)
grain = adptvgrnMod(grain, strength = 0.04, static = False, luma_scaling = 6, grain_chroma = False, size = 1.2, sharp = 80)
#endcard_splice = lvf.rfs(grain, deblock2, 34763)
#endcard_splice = core.std.FreezeFrames(endcard_splice, 34764, 34882, 34763)
out = depth(grain, 10)


###WORKRAW
wr = dbs.dumb3kdb(ed, 17, [48, 55, 55], 0)
wr = HQDeringmod(wr, mrad = 3, msmooth = 1, mthr = 20, nrmode = 1)
wr = FineDehalo(wr, rx = 1.7, darkstr = 0, brightstr = 0.7)
wr = adptvgrnMod(wr, strength = 0.07, static = True, luma_scaling = 2, grain_chroma = False, size = 1.2, sharp = 80)
wr = depth(wr, 10)

out.set_output()

#ed.set_output(1)
#core.std.ShufflePlanes([aa1, borders2], [0, 1, 2], vs.YUV).set_output(2)
#core.std.ShufflePlanes([aastrong, borders2], [0, 1, 2], vs.YUV).set_output(3)
#merge1.set_output(4)
#dering1.set_output(5)
#chromafix.set_output(6)
#chromafix.set_output(7)