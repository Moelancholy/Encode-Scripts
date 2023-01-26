import vapoursynth as vs
core = vs.core

core.max_cache_size = 30*1024
core.num_threads = 10


from awsmfunc import bbmod
from havsfunc import SMDegrain
from jvsfunc import retinex_edgemask
from rektlvls import rektlvls
from vodesfunc import Waifu2x_Doubler, NNEDI_Doubler
from vsaa import Eedi3SR, upscaled_sraa
from vsdeband import F3kdb, adaptive_grain
from vskernels import Bicubic, BicubicDidee
from vsmasktools import diff_rescale, MinMax, diff_creditless
from vsscale import SSIM
from vstools import replace_ranges, get_w
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
op_start = 720
op_end = 2877
op_len = op_end - op_start + 1

ed_start = 31554
ed_end = 33711
ed_len = ed_end - ed_start + 1


strong_opaa_ranges = [
    (81, 201), (239, 356), (779, 833), (1394, 1465),
    (1614, 1702)
]


native_ranges = [
    (2898, 3031)
]


###SOURCE/FIX BORDERS
src = core.lsmas.LWLibavSource(r"B:\サーバント×サービス\[BDMV][アニメ][140122] サーバント×サービス 6(完全生産限定版)\BDROM\BDMV\STREAM\00000.m2ts")
src = rektlvls(depth(src, 16), colnum = [0,1,1918,1919], colval = [3,1,1,3], rownum = [0,1,1078,1079], rowval = [2,1,1,2])
src = depth(bbmod(src, 2, 2, 2, 2, thresh = 4, blur = 999, u = True, v = True), 8)

src_y = get_y(depth(src, 32))

ncop = core.lsmas.LWLibavSource(r"B:\サーバント×サービス\[BDMV][アニメ][130821] サーバント×サービス 1(完全生産限定版)\BDMV\STREAM\00001.m2ts")[24:2181]
ncop = ncop[:op_len]

nced = core.lsmas.LWLibavSource(r"B:\サーバント×サービス\[BDMV][アニメ][130821] サーバント×サービス 1(完全生産限定版)\BDMV\STREAM\00002.m2ts")[24:2170]
nced = nced[:ed_len]
nced_y = get_y(depth(nced, 32))

op = src[op_start:op_end+1]
ed = src[ed_start:ed_end+1]

opmask = diff_creditless(op, ncop, thr = 0.9, expand = 5).std.Convolution([1]*25)
opmask = opmask.std.Crop(10, 2, 2, 2).std.AddBorders(10,2,2,2)
edmask = diff_creditless(ed, nced, thr = 0.8, expand = 5).std.Convolution([1]*25)



##RESCALE + AA MAIN
descale = Bicubic().descale(src_y, 1280, 720)
upscale = Waifu2x_Doubler(tiles = 2).double(descale)

aa = eedi3s(upscale.std.Transpose(), gamma = 180)
aa = eedi3s(aa.std.Transpose(), gamma = 180)

downscale = SSIM(kernel = BicubicDidee).scale(upscale, 1920, 1080)
downscale_aa = SSIM(kernel = BicubicDidee).scale(aa, 1920, 1080)

aamask = core.std.Prewitt(depth(downscale, 16)).std.Binarize(90 << 8).std.Maximum().std.Convolution([1]*9)
aamasked = core.std.MaskedMerge(downscale, downscale_aa, depth(aamask, 32))

ed_masked = core.std.MaskedMerge(aamasked[ed_start:ed_end+1], get_y(depth(ed, 32)), depth(edmask, 32))
aa_main = aamasked[:ed_start] + ed_masked + aamasked[ed_end+1:]


##AA OP
op_y = get_y(depth(ncop, 16))
op_y2 = get_y(depth(op, 16))

up = NNEDI_Doubler().double(op_y)
down1 = SSIM(kernel=BicubicDidee).scale(op_y, get_w(1620), 1620)

opaa1 = eedi3s(down1.std.Transpose(), gamma = 60)
opaa1 = eedi3s(opaa1.std.Transpose(), gamma = 60)
opaa_weak = SSIM(kernel=BicubicDidee).scale(opaa1, 1920, 1080)

opaa_strong = upscaled_sraa(op_y, rfactor = 1.5, aafunc=Eedi3SR(alpha = 0.25, beta= 0.5, gamma = 120))
opaa_mask = op_y.std.Prewitt().std.Binarize(40 << 8).std.Maximum().std.Convolution([1]*9)

opaa = replace_ranges(opaa_weak, opaa_strong, strong_opaa_ranges)
opaa = core.std.MaskedMerge(op_y, opaa, opaa_mask)
opaa = core.std.MaskedMerge(opaa, op_y2, depth(opmask, 16))
opaa = depth(opaa, 32)

aa_full = aa_main[:op_start] + opaa + opaa[2156] + aa_main[op_end+1:]
#aa_full = aa_main[:op_start] + opaa + aa_main[op_end+1:]
aa_full = depth(aa_full, 16)

descalemask = depth(diff_rescale(src, 720, thr = 55, expand=4), 16)
nativemasked = core.std.MaskedMerge(aa_full, depth(src_y, 16), descalemask)
nativespliced = replace_ranges(aa_full, nativemasked, native_ranges)


###DENOISE
denoise_y = SMDegrain(nativespliced, tr = 2, thSAD = 60, RefineMotion = True, prefilter = 2)
denoise_merge1 = core.std.ShufflePlanes([denoise_y, depth(src, 16)], [0,1,2], vs.YUV)

cdenpre = SMDegrain(denoise_merge1, tr = 2, thSAD=1, thSADC = 60, RefineMotion=True, chroma = True)
cden = core.knlm.KNLMeansCL(denoise_merge1, d = 4, a = 4, s = 3, h = 0.25, channels = 'uv', rclip = cdenpre)


###DEBAND
linemask = retinex_edgemask(depth(src_y, 16), brz = 9000)
texturemask = texture_mask(denoise_y, 1)
debandmask = core.akarin.Expr([linemask, texturemask], 'x 1.5 y * +')

deband = F3kdb.deband(cden, 16, [35, 30, 30])
debandmasked = core.std.MaskedMerge(deband, cden, debandmask)


###GRAIN
grain1 = core.noise.Add(debandmasked, type = 3, var = 2, seed = 5, xsize = 3, ysize = 3, constant = True)
grain1 = core.noise.Add(grain1, type = 3, var = 3, seed = 5, xsize = 3, ysize = 3)
grain2 = core.noise.Add(debandmasked, type = 3, var = 4, seed = 5, xsize = 3, ysize = 3, constant = True)

grainmask = adaptive_grain(src_y, luma_scaling = 3, show_mask = True)
grain = core.std.MaskedMerge(grain2, grain1, depth(grainmask, 16))


###OUTPUT
out = depth(grain, 10)
out.set_output()


#src.set_output()
#out.set_output(2)
#grain1.set_output(3)
#grain2.set_output(4)
#grain3.set_output(5)
#debandmask.set_output(6)
#denoise_y5.set_output(7)
#compare.set_output(5)



#deband
#regrain