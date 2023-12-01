from havsfunc import SMDegrain
from jvsfunc import retinex_edgemask
from vodesfunc import NNEDI_Doubler, adaptive_grain
from vsdeband import F3kdb
from vsdehalo import fine_dehalo, dehalo_sigma
from vsdenoise import dpir, Prefilter
from vskernels import BicubicDidee
from vsmasktools import MinMax
from vsrgtools import lehmer_diff_merge
from vsscale import SSIM
from vstools import vs, core, depth, get_y, join, iterate

core.set_affinity(range(18, 24), 30000)


###CUSTOM FUNCS
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
opranges = [
    (1560, 3716)
]

edranges = [
    (32129, 34286)
]


###SOURCES
cbr = depth(core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Mobile Suit Gundam - The Witch from Mercury - 24 (Amazon dAnime CBR 1080p).mkv"), 16)
vbr = depth(core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Mobile Suit Gundam - The Witch from Mercury - 24 (Amazon dAnime VBR 1080p).mkv"), 16)

crsrc = depth(core.lsmas.LWLibavSource(r"B:\[SubsPlease] Mobile Suit Gundam - The Witch from Mercury - 24 (1080p) [39DDE161].mkv"), 16)
cr = cbr[:192] + crsrc


###INTRO
lehmer = lehmer_diff_merge([cbr, vbr])
src = lehmer[:192] + crsrc

fixintro =  dpir(src, strength = 25, cuda = 'trt', tiles = 2, fp16 = True, matrix = 1)
introfixed = fixintro[:192] + crsrc


###AA
luma = get_y(introfixed)
double = NNEDI_Doubler(opencl=False).double(luma)
mid = SSIM(kernel=BicubicDidee()).scale(double, 2880, 1620)

aa2 = eedi3s(mid.std.Transpose(), gamma = 160)
aa2 = eedi3s(aa2.std.Transpose(), gamma = 160)
aa2 = SSIM(kernel=BicubicDidee()).scale(aa2, 1920, 1080)

aamask = luma.std.Prewitt().std.Binarize(69 << 8).std.Maximum().std.BoxBlur()
aamasked = core.std.MaskedMerge(luma, aa2, aamask)


###DEHALO LUMA
doubledh = aamasked.znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3).std.Transpose().znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3).std.Transpose()
dehalomask = fine_dehalo(doubledh, show_mask=True).resize.Bicubic(1920, 1080)
dehalomask2 = dehalomask.std.BoxBlur()

dehalo2 = dehalo_sigma(doubledh, brightstr = 1, blur_func=Prefilter.GAUSSBLUR2(sigma = 1.55)).resize.Point(1920, 1080)
dehalo = core.std.MaskedMerge(aamasked, dehalo2, dehalomask2)
dh32 = depth(dehalo, 32)


###DENOISE LUMA
preden = depth(SMDegrain(dehalo, tr = 2, thSAD = 120, RefineMotion = True, prefilter = 2), 32)
denoise_y = core.bm3dcuda_rtc.BM3D(clip = dh32, ref = preden, sigma = 1.00, radius = 3, block_step = 6, ps_range = 5).bm3d.VAggregate(radius = 3)
denoise_y = core.bm3dcuda_rtc.BM3D(clip = dh32, ref = depth(denoise_y, 32), sigma = 1.00, radius = 3, block_step = 5, ps_range = 6).bm3d.VAggregate(radius = 3)


###FIX CHROMA
dnmerge = join(denoise_y, introfixed)

dpir1 = dpir(dnmerge, strength = 25, cuda = 'trt', tiles = 2, fp16 = True, matrix = 1)
dpir1 = join(denoise_y, dpir1)

cden = core.knlm.KNLMeansCL(dpir1, d = 4, a = 4, s = 3, h = 0.25, channels = 'uv')


###DEBAND
linemask = retinex_edgemask(denoise_y)
tmask = texture_mask(denoise_y, 1)
debandmask = core.std.Expr([tmask, linemask], "x y max")

deband = F3kdb.deband(cden, 18, [35*4, 25*4, 25*4], 0)
deband = core.std.MaskedMerge(deband, cden, debandmask)


###GRAIN
grain = adaptive_grain(deband, [2.3, 0], seed = 5)
out = depth(grain, 10)


##WORKRAW
wrdenoise = core.knlm.KNLMeansCL(src, d = 4, a = 4, s = 3, h = 0.4, channels = 'y')
wrdenoise = core.knlm.KNLMeansCL(src, d = 4, a = 4, s = 3, h = 0.25, channels = 'uv')
wrdeband = F3kdb.deband(wrdenoise, 18, [35*4, 25*4, 25*4], 0)
wrgrain = adaptive_grain(wrdeband, [2.3, 0], seed = 5)
wrdone = depth(wrgrain, 10)


###OUTPUT
src.set_output()
#deband.set_output(2)
#grain1.set_output(3)
#grain2.set_output(4)
#grain3.set_output(5)
#deband3.set_output(6)

#6519 banding