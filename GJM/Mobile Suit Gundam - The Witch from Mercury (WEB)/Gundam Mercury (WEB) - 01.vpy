import vapoursynth as vs
core = vs.core


import awsmfunc as awf
import debandshit as dbs
import lvsfunc as lvf
import vardefunc as vdf
import vardefunc.mask as mask

from adptvgrnMod import adptvgrnMod
from havsfunc import FineDehalo, SMDegrain, HQDeringmod
from kagefunc import retinex_edgemask
from lvsfunc.recon import _Regress, _ReconstructMulti, chroma_reconstruct
from muvsfunc import SSIM_downsample
from rekt import rektlvls
from vardefunc.scale import to_444, nnedi3_upscale, nnedi3cl_double
from vsdpir import DPIR
from vsutil import get_depth, get_y, depth, plane, iterate, join
from xvs import mwcfix


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
    (318, 506), (855, 881), (1386, 1497),
    (1596, 1684), (2033, 2134), (2659, 2709),
    (3748, 3781), (4233, 4307), (5368, 5487),
    (5956, 6021), (9140, 9222), (9364, 9399),
    (18951, 19008), (21683, 21713), (24737, 24844),
    (25618, 25669), (26116, 26154), (26205, 26247),
    (26268, 26314), (26467, 26500), (26573, 26698),
    (28491, 28562), (29715, 29756), (31508, 31529),
]


dering_ranges = [
    (2808, 3080), (3818, 4084), (9736, 10053),
    (10446, 10548), (10682, 10795), (18951, 19008),
    (25957, 25995), (26079, 26115), (26155, 26379),
    (26803, 26880), (28635, 28636), (29072, 29145),
    (29170, 29319), (29516, 29695), (29757, 29858),
    (29877, 29912), 
]


cr_signs_ranges = [
    (6738, 6833), (13809, 13916), (22635, 22718),
    (34523, 34762)
]


ed_ranges = [
    (32366, 34522)
]


endcard_ranges = [
    (34763, 34882)
]


###SOURCES
cbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Mobile Suit Gundam - The Witch from Mercury - 01 (Amazon dAnime CBR 1080p) (1).mkv")
vbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Mobile Suit Gundam - The Witch from Mercury - 01 (Amazon dAnime VBR 1080p) (1).mkv")
disney = depth(core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Mobile Suit Gundam - The Witch from Mercury - 01 (Disney+ 1080p).mkv"), 16)[24:]
cr = depth(core.lsmas.LWLibavSource(r"E:\Anime\[SubsPlease] Mobile Suit Gundam - The Witch from Mercury - 01 (1080p) [FAD0DC07].mkv"), 16)

blank = core.std.BlankClip(disney)
disney = blank[:192] + disney
cr = blank[:192] + cr

src = lvf.rfs(disney, cr, cr_signs_ranges)

cbr16 = depth(cbr, 16)
ed = cbr16[:192] + src[192:]


##FIX BORDERS
borders1 = rektlvls(ed, colnum = [0, 1919], colval = [43, 43])
borders2 = awf.bbmod(borders1, left = 1, right = 1, thresh = 5, blur = 999, u = True, v = True)
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
denoise_y = core.bm3dcuda.BM3D(clip = dh32, ref = preden, sigma = 0.90, radius = 1).bm3d.VAggregate(radius = 1)
denoise_y = core.bm3dcuda.BM3D(clip = dh32, ref = depth(denoise_y, 32), sigma = 0.90, radius = 1).bm3d.VAggregate(radius = 1)


###DERING
merge1 = core.std.ShufflePlanes([denoise_y, borders2], [0, 1, 2], vs.YUV)
merge1 = deblock1[:192] + merge1[192:]
dering1 = HQDeringmod(merge1, mrad = 3, msmooth = 1, mthr = 20, nrmode = 1)
dering = lvf.rfs(merge1, dering1, dering_ranges)


###CHROMA
chromadn = ccdmod(dering, 4)
linemask = retinex_edgemask(aa_masked)
dn_merged = core.std.MaskedMerge(chromadn, dering, linemask)
up = plane(dn_merged, 1)
vp = plane(dn_merged, 2)
vpaa = nnedi3cl_double(vp)
vpaa = eedi3s(vpaa.std.Transpose(), gamma = 150)
vpaa = eedi3s(vpaa.std.Transpose(), gamma = 150).resize.Bicubic(960, 540)
luma = get_y(dn_merged)
ups = core.resize.Bicubic(up, src_left=-0.25)
vps = core.resize.Bicubic(vp, src_left=-0.25)
d444 = join([luma, ups, vps])
drgb = core.resize.Bicubic(d444, format = vs.RGBS)
dd = DPIR(drgb, strength = 10, task = 'deblock', device_type = 'cpu', fp16 = False, tile_x = 540, tile_y = 270, tile_pad=10).resize.Bicubic(format = vs.YUV444P16, matrix = 1)
vpd = plane(dd, 2)

luma_m = luma.resize.Point(960, src_left=-1).resize.Bicubic(960, 540)
luma_m = luma_m.nnedi3cl.NNEDI3CL(1, 1, 1, 0, 0, 4, 2).resize.Bicubic(src_top=-0.5)
u_m = up.nnedi3cl.NNEDI3CL(1, 1, 1, 0, 0, 4, 2).resize.Bicubic(src_top=-0.5)
v_m = vp.nnedi3cl.NNEDI3CL(1, 1, 1, 0, 0, 4, 2).resize.Bicubic(src_top=-0.5)

yfix = core.std.MakeDiff(luma, luma_m)
yu, yv = _Regress(luma_m, u_m, u_m, radius = 2)

u_fix = _ReconstructMulti(yfix, yu, radius = 2)
u_r = core.std.MergeDiff(u_m, u_fix)

v_fix = _ReconstructMulti(yfix, yv, radius = 2)
v_r = core.std.MergeDiff(v_m, v_fix)

merged_recon = join([luma, u_r, v_r])
test1 = chroma_reconstruct(dn_merged, i444 = True)
test2 = test1.resize.Bicubic(format = vs.RGBS)
test2 = DPIR(test2, strength = 10, task = 'deblock', device_type = 'cpu').resize.Bicubic(format = vs.YUV444P16, matrix = 1)
test2 = core.std.ShufflePlanes([dn_merged, test2], [0, 1, 2], vs.YUV)

u_recon = plane(test2, 1)
v_recon = plane(test2, 2)

v_dh1 = FineDehalo(v_recon, rx = 1.7, darkstr=0, brightstr=1)
v_double = v_recon.znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3).std.Transpose().znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3).std.Transpose()
v_dh2 = FineDehalo(v_double, rx = 2.2, darkstr=0, brightstr=1).resize.Point(1920, 1080)
v_dh = core.std.Expr([v_dh1, v_dh2], "x y min")

test3 = core.std.ShufflePlanes([dn_merged, u_recon, v_dh], [0, 0, 0], vs.YUV)

###DEBAND
deband = dbs.dumb3kdb(dn_merged, 17, [28, 32, 32], 0)
deband = adptvgrnMod(deband, strength = 0.04, size = 1.2, sharp = 80, luma_scaling = 2, static = True, grain_chroma = False)
deband = adptvgrnMod(deband, strength = 0.10, size = 1.2, sharp = 60, luma_scaling = 8, static = False, grain_chroma = False)

strongdeband = dbs.dumb3kdb(dn_merged, 17, [48, 50, 50], 0)
strongdeband = adptvgrnMod(deband, strength = 0.05, size = 1.2, sharp = 80, luma_scaling = 2, static = True, grain_chroma = False)
strongdeband = adptvgrnMod(deband, strength = 0.11, size = 1.2, sharp = 60, luma_scaling = 8, static = False, grain_chroma = False)

tmask = texture_mask(get_y(dn_merged), 1)
debandmask = core.std.Expr([tmask, linemask], "x y max")
debanded = lvf.rfs(deband, strongdeband, cr_signs_ranges)
deband_masked = core.std.MaskedMerge(debanded, dn_merged, debandmask)


###OUTPUT
ed_splice = lvf.rfs(deband_masked, ed, ed_ranges)
grain = adptvgrnMod(deband_masked, strength = 0.03, static = True, luma_scaling = 2, grain_chroma = False, size = 1.2, sharp = 80)
grain = adptvgrnMod(grain, strength = 0.05, static = False, luma_scaling = 6, grain_chroma = False, size = 1.2, sharp = 80)
endcard_splice = lvf.rfs(grain, deblock2, 34763)
endcard_splice = core.std.FreezeFrames(endcard_splice, 34764, 34882, 34763)
out = depth(endcard_splice, 10)

#out.set_output()

dn_merged.set_output(1)
test2.set_output(2)
test3.set_output(3)
plane(test3, 1).set_output(4)
plane(test3, 2).set_output(5)
#v_dh2.set_output(6)