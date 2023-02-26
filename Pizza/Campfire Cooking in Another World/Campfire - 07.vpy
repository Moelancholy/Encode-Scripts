from vstools import replace_ranges, vs, core

core.set_affinity(range(12, 24), 40000)

from havsfunc import SMDegrain
from jvsfunc import ccdmod, retinex_edgemask
from vodesfunc import Waifu2x_Doubler
from vsdeband import F3kdb, adaptive_grain
from vskernels import Bicubic, BicubicDidee, Bilinear
from vsmasktools import diff_rescale, MinMax
from vsscale import SSIM
from vstools import replace_ranges
from vsutil import depth, get_y, iterate



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


def texture_mask(clip, range):
    ed_gray = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    rmask = MinMax(range).edgemask(clip, lthr=0, multi=1.00)
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
opstart = 3213
edstart = 29154

native_ranges = [
    (opstart, opstart+2307), (30263, 32487), (34316, 34403)
]




###SOURCE
src = core.lsmas.LWLibavSource(r"B:\[SubsPlease] Tondemo Skill de Isekai Hourou Meshi - 07 (1080p) [C749639C].mkv")
amzn = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Tondemo Skill de Isekai Hourou Meshi - 07 (Amazon Prime VBR 1080p).mkv")
src_y = get_y(depth(src, 16))


##RESCALE + AA
descale = Bicubic(0, 1/2).descale(src_y, 1500, 844)
descale2 = Bilinear().descale(src_y, 1500, 844)

waifu = Waifu2x_Doubler(tiles=2).double(descale)
rescale = SSIM(kernel=BicubicDidee).scale(waifu, 1920, 1080)

aa1 = eedi3s(waifu.std.Transpose(), gamma = 70)
aa1 = eedi3s(aa1.std.Transpose(), gamma = 70)
aa1 = SSIM(kernel=BicubicDidee).scale(aa1, 1920, 1080)

aamask = rescale.std.Prewitt().std.Binarize(69 << 8).std.Maximum().std.Convolution([1]*9)
aa = core.std.MaskedMerge(rescale, aa1, aamask)

descalemask = depth(diff_rescale(src, 844, thr = 55, expand=4), 16)
descale_masked = core.std.MaskedMerge(aa, src_y, descalemask)

rescale_final = replace_ranges(aa, descale_masked, native_ranges)


###DENOISE
preden = SMDegrain(rescale_final, tr = 2, thSAD = 150, RefineMotion = True, prefilter = 2)
preden32 = depth(preden, 32)

denoise_y = core.bm3dcuda_rtc.BM3D(clip = depth(rescale_final, 32), ref = preden32, sigma = 0.80, radius = 2).bm3d.VAggregate(radius = 2)
denoise_y = core.bm3dcuda_rtc.BM3D(clip = depth(rescale_final, 32), ref = depth(denoise_y, 32), sigma = 0.80, radius = 2).bm3d.VAggregate(radius = 2)

cden = ccdmod(depth(src, 16), 4.5)
cden = core.std.ShufflePlanes([denoise_y, cden], [0, 1, 2], vs.YUV)


###DEBAND
linemask = retinex_edgemask(src_y)
texturemask = texture_mask(denoise_y, 1)
debandmask = core.akarin.Expr([linemask, texturemask], 'x 1.5 y * +')

deband = F3kdb.deband(cden, 16, [36, 40, 40])
deband = core.std.MaskedMerge(deband, cden, debandmask)

###OUTPUT
grainmask = adaptive_grain(src_y, luma_scaling = 6, show_mask = True)

grain = core.noise.Add(deband, type = 3, var = 3, seed = 5)
grain = core.std.MaskedMerge(deband, grain, grainmask)
out = depth(grain, 10)


out.set_output()
#src.set_output(2)
#amzn.set_output(3)
#descalemask.set_output(4)
#grain3.set_output(5)
#grainmask.set_output(6)
#cden2.set_output(4)
#compare.set_output(5)